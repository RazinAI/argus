#include "rocm.hxx"
#include "logger.hxx"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <migraphx/migraphx.hpp>
#include <sstream>
#include <vector>

namespace {

// MIGraphX compiles for fixed shapes, so the dynamic YOLO export is pinned to
// [batch, 3, 640, 640] (uint8 models: [batch, 640, 640, 3]) at build time.
// Ultralytics names the input "images".
constexpr const char *kInputName = "images";
constexpr std::size_t kInputSize = 640;

std::string lengths_string(const std::vector<std::size_t> &lengths) {
  std::ostringstream out;
  for (std::size_t i = 0; i < lengths.size(); i++) {
    if (i > 0)
      out << "x";
    out << lengths[i];
  }
  return out.str();
}

std::size_t product(const std::vector<std::size_t> &lengths) {
  std::size_t elements = 1;
  for (std::size_t length : lengths)
    elements *= length;
  return elements;
}

class RocmEngine : public Engine {
public:
  bool infer(int slot, const void *const *cpu_inputs, float *cpu_output,
             int batch_size) override;

  migraphx::program program;
  std::string inputName;
  migraphx::shape inputShape;
  // Host buffer sized for the compiled batch. Partial batches are copied into
  // the front; leftover rows hold stale frames whose outputs are ignored.
  std::vector<unsigned char> inputBuffer;
};

bool RocmEngine::infer(int, const void *const *cpu_inputs,
                      float *cpu_output, int batch_size) {
  if (batch_size < 1 || batch_size > maxBatchSize) {
    log_msg(LOG_ERROR, "engine",
            "invalid inference batch size=" + std::to_string(batch_size) +
                " max=" + std::to_string(maxBatchSize));
    return false;
  }

  std::size_t frame_bytes = input_frame_bytes();
  for (int i = 0; i < batch_size; i++)
    std::memcpy(inputBuffer.data() + i * frame_bytes, cpu_inputs[i],
                frame_bytes);

  try {
    migraphx::program_parameters params;
    params.add(inputName.c_str(),
               migraphx::argument(inputShape, inputBuffer.data()));

    // offload_copy was enabled at compile time, so MIGraphX copies the input
    // to the GPU and the outputs back to host memory inside eval().
    migraphx::arguments outputs = program.eval(params);
    migraphx::argument output = outputs[0];

    std::size_t bytes = batch_size * outputElementsPerFrame * sizeof(float);
    std::memcpy(cpu_output, output.data(), bytes);
  } catch (const std::exception &err) {
    log_msg(LOG_ERROR, "engine",
            std::string("MIGraphX inference failed: ") + err.what());
    return false;
  }

  return true;
}

} // namespace

namespace rocm {

bool build_engine_file(const char *onnx_path, const char *engine_path,
                       int batch_size) {
  if (!create_parent_directory(engine_path))
    return false;

  try {
    // A float model is planar [batch, 3, 640, 640]; a uint8 one (add_nms.py
    // --uint8-input) is interleaved [batch, 640, 640, 3]. Parse once with
    // defaults just to read the input type.
    bool uint8_input =
        migraphx::parse_onnx(onnx_path).get_parameter_shapes()[kInputName]
            .type() == migraphx_shape_uint8_type;
    std::size_t batch = static_cast<std::size_t>(batch_size);
    migraphx::onnx_options options;
    if (uint8_input)
      options.set_input_parameter_shape(kInputName,
                                        {batch, kInputSize, kInputSize, 3});
    else
      options.set_input_parameter_shape(kInputName,
                                        {batch, 3, kInputSize, kInputSize});

    log_msg(LOG_INFO, "engine",
            std::string("ONNX parse started for ") + onnx_path);
    auto start = std::chrono::steady_clock::now();
    migraphx::program program = migraphx::parse_onnx(onnx_path, options);
    log_duration(LOG_INFO, "engine", "ONNX parse", start);

    migraphx::compile_options compile_options;
    compile_options.set_offload_copy();

    log_msg(LOG_INFO, "engine", "MIGraphX gpu compile started");
    start = std::chrono::steady_clock::now();
    program.compile(migraphx::target("gpu"), compile_options);
    log_duration(LOG_INFO, "engine", "MIGraphX gpu compile", start);

    migraphx::save(program, engine_path);
  } catch (const std::exception &err) {
    log_msg(LOG_CRITICAL, "engine",
            std::string("MIGraphX build failed: ") + err.what());
    return false;
  }

  log_msg(LOG_INFO, "engine",
          std::string("saved MIGraphX program file: ") + engine_path);
  return true;
}

std::unique_ptr<Engine> load_engine(const char *engine_path, int batch_size) {
  auto e = std::make_unique<RocmEngine>();

  try {
    log_msg(LOG_INFO, "engine", "MIGraphX program load started");
    auto start = std::chrono::steady_clock::now();
    e->program = migraphx::load(engine_path);
    log_duration(LOG_INFO, "engine", "MIGraphX program load", start);

    migraphx::program_parameter_shapes params =
        e->program.get_parameter_shapes();
    for (const char *name : params.names()) {
      // With offload_copy the only parameters are the model inputs; names
      // starting with '#' would be MIGraphX-internal output buffers.
      if (name[0] == '#')
        continue;
      e->inputName = name;
      e->inputShape = params[name];
      break;
    }
    if (e->inputName.empty()) {
      log_msg(LOG_CRITICAL, "engine", "MIGraphX program has no input");
      return nullptr;
    }

    migraphx::shapes outputs = e->program.get_output_shapes();
    if (outputs.size() == 0) {
      log_msg(LOG_CRITICAL, "engine", "MIGraphX program has no output");
      return nullptr;
    }
    migraphx::shape output_shape = outputs[0];

    bool uint8_input = e->inputShape.type() == migraphx_shape_uint8_type;
    if ((!uint8_input && e->inputShape.type() != migraphx_shape_float_type) ||
        output_shape.type() != migraphx_shape_float_type) {
      log_msg(LOG_CRITICAL, "engine",
              "MIGraphX program input must be fp32 or uint8 and output fp32");
      return nullptr;
    }
    e->inputType = uint8_input ? InputType::Uint8 : InputType::Float32;

    std::vector<std::size_t> input_lengths = e->inputShape.lengths();
    std::vector<std::size_t> output_lengths = output_shape.lengths();
    int compiled_batch = static_cast<int>(input_lengths[0]);

    log_msg(LOG_INFO, "engine",
            "input " + e->inputName + "=" + lengths_string(input_lengths));
    log_msg(LOG_INFO, "engine", "output=" + lengths_string(output_lengths));

    if (batch_size > compiled_batch) {
      log_msg(LOG_CRITICAL, "engine",
              "requested batch=" + std::to_string(batch_size) +
                  " exceeds MIGraphX compiled batch=" +
                  std::to_string(compiled_batch) +
                  "; rebuild with a larger --batch");
      return nullptr;
    }
    if (batch_size < compiled_batch)
      log_msg(LOG_WARNING, "engine",
              "MIGraphX runs every inference at compiled batch=" +
                  std::to_string(compiled_batch) +
                  "; rebuild with --batch " + std::to_string(batch_size) +
                  " to avoid padded work");

    e->maxBatchSize = batch_size;
    e->inputElementsPerFrame = product(input_lengths) / compiled_batch;
    e->outputElementsPerFrame = product(output_lengths) / output_lengths[0];
    e->outputDims.assign(output_lengths.begin() + 1, output_lengths.end());
    e->inputBuffer.assign(compiled_batch * e->input_frame_bytes(), 0);
  } catch (const std::exception &err) {
    log_msg(LOG_CRITICAL, "engine",
            std::string("MIGraphX load failed: ") + err.what());
    return nullptr;
  }

  return e;
}

} // namespace rocm

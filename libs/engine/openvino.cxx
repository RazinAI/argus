#include "openvino.hxx"
#include "logger.hxx"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

namespace {

constexpr std::int64_t kInputSize = 640;

// Prefer an Intel GPU when one is present, otherwise run on the CPU. The GPU
// plugin also lists other vendors' OpenCL devices (e.g. NVIDIA), which it
// does not support, so only GPUs named Intel count. The exported engine file
// is tied to the device it was compiled for, so build and load must pick the
// same one.
std::string pick_device(ov::Core &core) {
  for (const std::string &device : core.get_available_devices()) {
    if (device.rfind("GPU", 0) != 0)
      continue;
    std::string name = core.get_property(device, ov::device::full_name);
    if (name.find("Intel") != std::string::npos)
      return device;
  }
  return "CPU";
}

std::size_t elements_after_batch(const ov::PartialShape &shape) {
  std::size_t elements = 1;
  for (std::size_t i = 1; i < shape.size(); i++)
    elements *= static_cast<std::size_t>(shape[i].get_length());
  return elements;
}

class OpenvinoEngine : public Engine {
public:
  bool infer(int slot, const void *const *cpu_inputs, float *cpu_output,
             int batch_size) override;

  ov::Core core;
  ov::CompiledModel compiled;
  // In THROUGHPUT mode the CPU is split into several streams, each running
  // its own request. infer() splits the batch across these requests.
  std::vector<ov::InferRequest> requests;
  // OpenVINO reads a batch from one contiguous buffer, so frames are copied
  // together here first.
  std::vector<unsigned char> batchInput;
  // Input shape of one frame, batch excluded.
  ov::Shape inputFrameShape;
};

bool OpenvinoEngine::infer(int, const void *const *cpu_inputs,
                          float *cpu_output, int batch_size) {
  if (batch_size < 1 || batch_size > maxBatchSize) {
    log_msg(LOG_ERROR, "engine",
            "invalid inference batch size=" + std::to_string(batch_size) +
                " max=" + std::to_string(maxBatchSize));
    return false;
  }

  std::size_t frame_bytes = input_frame_bytes();
  batchInput.resize(batch_size * frame_bytes);
  for (int i = 0; i < batch_size; i++)
    std::memcpy(batchInput.data() + i * frame_bytes, cpu_inputs[i],
                frame_bytes);
  ov::element::Type input_type =
      inputType == InputType::Uint8 ? ov::element::u8 : ov::element::f32;

  try {
    // Split the batch as evenly as possible, one slice per request, and run
    // the slices at the same time.
    int slices = std::min<int>(batch_size, requests.size());
    std::vector<int> first(slices), count(slices);
    for (int i = 0, frame = 0; i < slices; i++) {
      count[i] = batch_size / slices + (i < batch_size % slices ? 1 : 0);
      first[i] = frame;
      frame += count[i];
    }

    for (int i = 0; i < slices; i++) {
      // Wrap the batch buffer instead of copying it again.
      ov::Shape shape{static_cast<std::size_t>(count[i])};
      shape.insert(shape.end(), inputFrameShape.begin(),
                   inputFrameShape.end());
      ov::Tensor input(input_type, shape,
                       batchInput.data() + first[i] * frame_bytes);
      requests[i].set_input_tensor(input);
      requests[i].start_async();
    }

    for (int i = 0; i < slices; i++) {
      requests[i].wait();
      ov::Tensor output = requests[i].get_output_tensor();
      std::size_t bytes = count[i] * outputElementsPerFrame * sizeof(float);
      if (output.get_byte_size() < bytes) {
        log_msg(LOG_ERROR, "engine", "OpenVINO output is smaller than expected");
        return false;
      }
      std::memcpy(cpu_output + first[i] * outputElementsPerFrame,
                  output.data(), bytes);
    }
  } catch (const std::exception &err) {
    log_msg(LOG_ERROR, "engine",
            std::string("OpenVINO inference failed: ") + err.what());
    return false;
  }

  return true;
}

} // namespace

namespace openvino {

bool build_engine_file(const char *onnx_path, const char *engine_path,
                       int batch_size) {
  if (!create_parent_directory(engine_path))
    return false;

  try {
    ov::Core core;
    std::string device = pick_device(core);
    log_msg(LOG_INFO, "engine", "OpenVINO version " +
                                    std::string(ov::get_openvino_version()
                                                    .buildNumber) +
                                    " device " + device);

    log_msg(LOG_INFO, "engine",
            std::string("ONNX read started for ") + onnx_path);
    auto start = std::chrono::steady_clock::now();
    std::shared_ptr<ov::Model> model = core.read_model(onnx_path);
    log_duration(LOG_INFO, "engine", "ONNX read", start);

    // Allow any batch from 1 to batch_size, like the TensorRT optimization
    // profile, and fix any other dynamic dimension (height/width) to 640.
    ov::PartialShape input_shape = model->input().get_partial_shape();
    input_shape[0] = ov::Dimension(1, batch_size);
    for (std::size_t i = 1; i < input_shape.size(); i++)
      if (input_shape[i].is_dynamic())
        input_shape[i] = kInputSize;
    model->reshape(input_shape);
    log_msg(LOG_INFO, "engine",
            "input " + model->input().get_any_name() + "=" +
                input_shape.to_string());

    log_msg(LOG_INFO, "engine", "OpenVINO compile started");
    start = std::chrono::steady_clock::now();
    ov::CompiledModel compiled = core.compile_model(model, device);
    log_duration(LOG_INFO, "engine", "OpenVINO compile", start);

    std::ofstream file(engine_path, std::ios::binary | std::ios::trunc);
    if (!file) {
      log_msg(LOG_CRITICAL, "engine",
              std::string("could not create OpenVINO engine file: ") +
                  engine_path);
      return false;
    }
    compiled.export_model(file);
    if (!file) {
      log_msg(LOG_CRITICAL, "engine", "could not write OpenVINO engine file");
      return false;
    }
  } catch (const std::exception &err) {
    log_msg(LOG_CRITICAL, "engine",
            std::string("OpenVINO build failed: ") + err.what());
    return false;
  }

  log_msg(LOG_INFO, "engine",
          std::string("saved OpenVINO engine file: ") + engine_path);
  return true;
}

std::unique_ptr<Engine> load_engine(const char *engine_path, int batch_size) {
  auto e = std::make_unique<OpenvinoEngine>();

  std::ifstream file(engine_path, std::ios::binary);
  if (!file) {
    log_msg(LOG_CRITICAL, "engine",
            std::string("could not open OpenVINO engine file: ") +
                engine_path);
    return nullptr;
  }

  try {
    std::string device = pick_device(e->core);
    // TEMPORARY experiment switch: ARGUS_OV_MODE=latency|throughput.
    const char *mode_env = std::getenv("ARGUS_OV_MODE");
    bool throughput = mode_env && std::string(mode_env) == "throughput";
    ov::hint::PerformanceMode mode =
        throughput ? ov::hint::PerformanceMode::THROUGHPUT
                   : ov::hint::PerformanceMode::LATENCY;
    log_msg(LOG_INFO, "engine",
            "OpenVINO import started on " + device + " mode " +
                (throughput ? "THROUGHPUT" : "LATENCY"));
    auto start = std::chrono::steady_clock::now();
    e->compiled = e->core.import_model(file, device,
                                       ov::hint::performance_mode(mode));
    log_duration(LOG_INFO, "engine", "OpenVINO import", start);

    if (e->compiled.inputs().size() != 1 ||
        e->compiled.outputs().size() != 1) {
      log_msg(LOG_CRITICAL, "engine",
              "OpenVINO model must have one input and one output");
      return nullptr;
    }

    ov::Output<const ov::Node> input = e->compiled.input();
    ov::Output<const ov::Node> output = e->compiled.output();
    ov::PartialShape input_shape = input.get_partial_shape();
    ov::PartialShape output_shape = output.get_partial_shape();
    log_msg(LOG_INFO, "engine",
            "input " + input.get_any_name() + "=" + input_shape.to_string() +
                " output " + output.get_any_name() + "=" +
                output_shape.to_string());

    ov::element::Type input_type = input.get_element_type();
    if ((input_type != ov::element::f32 && input_type != ov::element::u8) ||
        output.get_element_type() != ov::element::f32) {
      log_msg(LOG_CRITICAL, "engine",
              "OpenVINO model input must be fp32 or u8 and output fp32");
      return nullptr;
    }
    e->inputType =
        input_type == ov::element::u8 ? InputType::Uint8 : InputType::Float32;

    std::int64_t compiled_batch = input_shape[0].get_max_length();
    if (compiled_batch >= 0 && batch_size > compiled_batch) {
      log_msg(LOG_CRITICAL, "engine",
              "requested batch=" + std::to_string(batch_size) +
                  " exceeds OpenVINO compiled batch=" +
                  std::to_string(compiled_batch) +
                  "; rebuild with a larger --batch");
      return nullptr;
    }

    std::uint32_t request_count =
        e->compiled.get_property(ov::optimal_number_of_infer_requests);
    log_msg(LOG_INFO, "engine",
            "OpenVINO streams=" +
                std::to_string(e->compiled.get_property(ov::num_streams).num) +
                " requests=" + std::to_string(request_count));
    for (std::uint32_t i = 0; i < std::max<std::uint32_t>(request_count, 1);
         i++)
      e->requests.push_back(e->compiled.create_infer_request());
    e->maxBatchSize = batch_size;
    e->inputElementsPerFrame = elements_after_batch(input_shape);
    for (std::size_t i = 1; i < input_shape.size(); i++)
      e->inputFrameShape.push_back(
          static_cast<std::size_t>(input_shape[i].get_length()));
    e->outputElementsPerFrame = elements_after_batch(output_shape);
    for (std::size_t i = 1; i < output_shape.size(); i++)
      e->outputDims.push_back(
          static_cast<std::size_t>(output_shape[i].get_length()));
  } catch (const std::exception &err) {
    log_msg(LOG_CRITICAL, "engine",
            std::string("OpenVINO load failed: ") + err.what());
    return nullptr;
  }

  return e;
}

} // namespace openvino

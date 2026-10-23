#include "cuda.hxx"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.hxx"
#include "profiling.hxx"
#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using namespace nvonnxparser;

namespace {
class TRTLogger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      LogLevel level = severity <= Severity::kERROR ? LOG_ERROR : LOG_WARNING;
      log_msg(level, "tensorrt", msg);
    }
  }
} logger;

std::string dims_string(Dims dims) {
  std::ostringstream out;
  for (int i = 0; i < dims.nbDims; i++) {
    if (i > 0)
      out << "x";
    out << dims.d[i];
  }
  return out.str();
}

const char *data_type_name(DataType type) {
  switch (type) {
  case DataType::kFLOAT:
    return "fp32";
  case DataType::kHALF:
    return "fp16";
  case DataType::kINT8:
    return "int8";
  case DataType::kINT32:
    return "int32";
  case DataType::kBOOL:
    return "bool";
  case DataType::kUINT8:
    return "uint8";
  case DataType::kFP8:
    return "fp8";
  case DataType::kBF16:
    return "bf16";
  case DataType::kINT64:
    return "int64";
  case DataType::kINT4:
    return "int4";
  case DataType::kFP4:
    return "fp4";
  case DataType::kE8M0:
    return "e8m0";
  }
  return "unknown";
}

std::size_t elem_size(DataType t) {
  switch (t) {
  case DataType::kFLOAT:
    return 4;
  case DataType::kHALF:
    return 2;
  case DataType::kBF16:
    return 2;
  case DataType::kINT32:
    return 4;
  case DataType::kINT64:
    return 8;
  case DataType::kINT8:
    return 1;
  case DataType::kUINT8:
    return 1;
  case DataType::kBOOL:
    return 1;
  case DataType::kFP8:
    return 1;
  default:
    return 0;
  }
}


class CudaEngine : public Engine {
public:
  ~CudaEngine() override;
  bool infer(int slot, const void *const *cpu_inputs, float *cpu_output,
             int batch_size) override;

  // Pinned host memory, so the GPU copies frames straight from it instead of
  // the driver staging them through its own pinned buffer first.
  void *allocate_input_frame() override;
  void free_input_frame(void *frame) override;

  // Everything one in-flight batch needs: a context (its intermediate
  // tensors), GPU input/output buffers, and a stream. Work on different
  // streams can overlap, so one slot's upload runs during another's kernels.
  struct Slot {
    std::unique_ptr<IExecutionContext> context;
    std::vector<void *> gpuBuffers;
    void *inputBuffer = nullptr;
    void *outputBuffer = nullptr;
    cudaStream_t stream = nullptr;
  };

  std::unique_ptr<IRuntime> runtime;
  std::unique_ptr<ICudaEngine> engine;
  std::vector<Slot> slots;
  std::string inputName;
  std::string outputName;
};

// Two batches in flight: one uploading while the other computes.
constexpr int kInferSlots = 2;

// The batch is dynamic; any other dynamic dimension is the image height or
// width, fixed to 640 (planar float [-1,3,-1,-1] or uint8 [-1,640,640,3]).
Dims input_dims_for_batch(Dims dims, int batch) {
  dims.d[0] = batch;
  for (int i = 1; i < dims.nbDims; i++)
    if (dims.d[i] < 0)
      dims.d[i] = 640;
  return dims;
}

bool write_engine_file(const char *engine_path, const void *data,
                       std::size_t size) {
  if (!create_parent_directory(engine_path))
    return false;

  std::ofstream file(engine_path, std::ios::binary | std::ios::trunc);
  if (!file) {
    log_msg(LOG_WARNING, "engine",
            std::string("could not create TensorRT engine file: ") +
                engine_path);
    return false;
  }

  file.write(static_cast<const char *>(data),
             static_cast<std::streamsize>(size));
  if (!file) {
    log_msg(LOG_WARNING, "engine", "could not write TensorRT engine file");
    return false;
  }

  log_msg(LOG_INFO, "engine",
          std::string("saved TensorRT engine file: ") + engine_path);
  return true;
}

} // namespace

namespace cuda {

bool build_engine_file(const char *onnx_path, const char *engine_path,
                       int batch_size) {
  log_msg(LOG_INFO, "engine",
          std::string("creating TensorRT builder for ") + onnx_path);
  std::unique_ptr<IBuilder> builder{createInferBuilder(logger)};
  if (!builder) {
    log_msg(LOG_CRITICAL, "engine", "failed to create TensorRT builder");
    return false;
  }

  std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))};
  if (!network) {
    log_msg(LOG_CRITICAL, "engine", "failed to create TensorRT network");
    return false;
  }

  std::unique_ptr<IParser> parser{createParser(*network, logger)};
  if (!parser) {
    log_msg(LOG_CRITICAL, "engine", "failed to create ONNX parser");
    return false;
  }

  {
    log_msg(LOG_INFO, "engine", "ONNX parse started");
    auto start = std::chrono::steady_clock::now();
    bool parsed = parser->parseFromFile(
        onnx_path, static_cast<int32_t>(ILogger::Severity::kWARNING));
    log_duration(LOG_INFO, "engine", "ONNX parse", start);

    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
      log_msg(LOG_ERROR, "engine", parser->getError(i)->desc());

    if (!parsed || parser->getNbErrors() > 0) {
      log_msg(LOG_CRITICAL, "engine", "ONNX parse failed");
      return false;
    }
  }

  std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
  if (!config) {
    log_msg(LOG_CRITICAL, "engine", "failed to create builder config");
    return false;
  }

  IOptimizationProfile *profile = builder->createOptimizationProfile();
  if (!profile) {
    log_msg(LOG_CRITICAL, "engine", "failed to create optimization profile");
    return false;
  }

  log_msg(LOG_INFO, "engine", "configuring optimization profile");
  for (int32_t i = 0; i < network->getNbInputs(); i++) {
    ITensor *input = network->getInput(i);
    Dims input_dims = input->getDimensions();
    Dims min_dims = input_dims_for_batch(input_dims, 1);
    Dims opt_dims = input_dims_for_batch(input_dims, std::max(1, batch_size / 2));
    Dims max_dims = input_dims_for_batch(input_dims, batch_size);

    profile->setDimensions(input->getName(), OptProfileSelector::kMIN,
                           min_dims);
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT,
                           opt_dims);
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX,
                           max_dims);

    log_msg(LOG_INFO, "engine",
            std::string("input profile ") + input->getName() + " min=" +
                dims_string(min_dims) + " opt=" + dims_string(opt_dims) +
                " max=" + dims_string(max_dims));
  }
  config->addOptimizationProfile(profile);

  std::unique_ptr<IHostMemory> serialized_model;
  {
    log_msg(LOG_INFO, "engine", "TensorRT serialized network build started");
    auto start = std::chrono::steady_clock::now();
    serialized_model.reset(builder->buildSerializedNetwork(*network, *config));
    log_duration(LOG_INFO, "engine", "TensorRT serialized network build",
                 start);
  }
  if (!serialized_model) {
    log_msg(LOG_CRITICAL, "engine",
            "failed to build serialized TensorRT network");
    return false;
  }

  return write_engine_file(engine_path, serialized_model->data(),
                           serialized_model->size());
}

std::unique_ptr<Engine> load_engine(const char *engine_path, int batch_size) {
  // FFmpeg only shares the primary context (for NVENC) if it was created with
  // blocking sync, so set that before anything initializes it. Waiting
  // threads then sleep until the GPU is done instead of spinning on a core.
  cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  if (flags_err != cudaSuccess)
    log_msg(LOG_WARNING, "engine",
            std::string("could not set blocking sync: ") +
                cudaGetErrorString(flags_err));

  auto owner = std::make_unique<CudaEngine>();
  CudaEngine &e = *owner;
  e.maxBatchSize = batch_size;
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file) {
    log_msg(LOG_CRITICAL, "engine",
            std::string("could not open TensorRT engine file: ") + engine_path);
    return nullptr;
  }

  std::streamsize size = file.tellg();
  if (size <= 0) {
    log_msg(LOG_CRITICAL, "engine", "TensorRT engine file is empty");
    return nullptr;
  }

  std::vector<char> engine_data(static_cast<std::size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(engine_data.data(), size)) {
    log_msg(LOG_CRITICAL, "engine", "could not read TensorRT engine file");
    return nullptr;
  }

  log_msg(LOG_INFO, "engine",
          "loaded TensorRT engine file bytes=" +
              std::to_string(engine_data.size()));

  log_msg(LOG_INFO, "engine", "creating TensorRT runtime");
  e.runtime.reset(createInferRuntime(logger));
  if (!e.runtime) {
    log_msg(LOG_CRITICAL, "engine", "failed to create TensorRT runtime");
    return nullptr;
  }

  {
    log_msg(LOG_INFO, "engine", "TensorRT engine deserialize started");
    auto start = std::chrono::steady_clock::now();
    e.engine.reset(e.runtime->deserializeCudaEngine(engine_data.data(),
                                                    engine_data.size()));
    log_duration(LOG_INFO, "engine", "TensorRT engine deserialize", start);
  }
  if (!e.engine) {
    log_msg(LOG_CRITICAL, "engine", "failed to deserialize TensorRT engine");
    return nullptr;
  }

  e.slots.resize(kInferSlots);
  for (int s = 0; s < kInferSlots; s++) {
    CudaEngine::Slot &slot = e.slots[s];
    std::string slot_name = "slot " + std::to_string(s) + ": ";

    log_msg(LOG_INFO, "engine", slot_name + "creating execution context");
    slot.context.reset(e.engine->createExecutionContext());
    if (!slot.context) {
      log_msg(LOG_CRITICAL, "engine", "failed to create execution context");
      return nullptr;
    }

    for (int32_t i = 0; i < e.engine->getNbIOTensors(); i++) {
      auto const name = e.engine->getIOTensorName(i);
      if (e.engine->getTensorIOMode(name) != TensorIOMode::kINPUT)
        continue;

      Dims dims = input_dims_for_batch(e.engine->getTensorShape(name),
                                       batch_size);

      if (!slot.context->setInputShape(name, dims)) {
        log_msg(LOG_CRITICAL, "engine",
                std::string("failed to set input shape for ") + name);
        return nullptr;
      }
      log_msg(LOG_INFO, "engine",
              slot_name + "input shape " + name + "=" + dims_string(dims));
    }

    log_msg(LOG_INFO, "engine", slot_name + "allocating tensor buffers");
    slot.gpuBuffers.assign(e.engine->getNbIOTensors(), nullptr);
    for (int32_t i = 0; i < e.engine->getNbIOTensors(); i++) {
      auto const name = e.engine->getIOTensorName(i);
      Dims dims = slot.context->getTensorShape(name);
      uint64_t elements = 1;
      for (int j = 0; j < dims.nbDims; j++)
        elements *= dims.d[j];
      DataType type = e.engine->getTensorDataType(name);
      std::size_t bytes = elements * elem_size(type);
      cudaMalloc(&slot.gpuBuffers[i], bytes);
      slot.context->setTensorAddress(name, slot.gpuBuffers[i]);

      log_msg(LOG_INFO, "engine",
              slot_name + "tensor " + name + " shape=" + dims_string(dims) +
                  " type=" + data_type_name(type) +
                  " bytes=" + std::to_string(bytes));

      if (e.engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
        if (type != DataType::kFLOAT && type != DataType::kUINT8) {
          log_msg(LOG_CRITICAL, "engine",
                  std::string("unsupported input type ") +
                      data_type_name(type) + "; expected fp32 or uint8");
          return nullptr;
        }
        e.inputType =
            type == DataType::kUINT8 ? InputType::Uint8 : InputType::Float32;
        e.inputName = name;
        slot.inputBuffer = slot.gpuBuffers[i];
        e.inputElementsPerFrame = elements / batch_size;
      } else {
        e.outputName = name;
        slot.outputBuffer = slot.gpuBuffers[i];
        e.outputElementsPerFrame = elements / batch_size;
        e.outputDims.assign(dims.d + 1, dims.d + dims.nbDims);
      }
    }

    if (!slot.inputBuffer || !slot.outputBuffer) {
      log_msg(LOG_CRITICAL, "engine",
              "TensorRT input/output tensors not found");
      return nullptr;
    }

    // FFmpeg's NVENC uploads run on the legacy default stream in this same
    // context; a blocking stream would wait on each of them and vice versa.
    cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking);
    log_msg(LOG_INFO, "engine", slot_name + "CUDA stream created");
  }
  e.slotCount = kInferSlots;

  return owner;
}

} // namespace cuda

bool CudaEngine::infer(int slot_index, const void *const *cpu_inputs,
                       float *cpu_output, int batch_size) {
  if (slot_index < 0 || slot_index >= (int)slots.size()) {
    log_msg(LOG_ERROR, "engine",
            "invalid inference slot=" + std::to_string(slot_index));
    return false;
  }
  Slot &slot = slots[slot_index];

  if (batch_size < 1 || batch_size > maxBatchSize) {
    log_msg(LOG_ERROR, "engine",
            "invalid inference batch size=" + std::to_string(batch_size) +
                " max=" + std::to_string(maxBatchSize));
    return false;
  }

  if (inputName.empty() || outputName.empty()) {
    log_msg(LOG_ERROR, "engine", "TensorRT tensor names are not initialized");
    return false;
  }

  Dims input_dims = slot.context->getTensorShape(inputName.c_str());
  input_dims.d[0] = batch_size;
  if (!slot.context->setInputShape(inputName.c_str(), input_dims)) {
    log_msg(LOG_ERROR, "engine", "failed to set runtime input batch shape");
    return false;
  }

  std::size_t frame_bytes = input_frame_bytes();
  {
    // Frames are pinned, so these copies only queue DMA work and return.
    ProfileRange range(Stage::Infer, "copy in");
    for (int i = 0; i < batch_size; i++) {
      if (!cpu_inputs[i]) {
        log_msg(LOG_ERROR, "engine", "missing input frame in batch");
        return false;
      }
      cudaMemcpyAsync(static_cast<char *>(slot.inputBuffer) + i * frame_bytes,
                      cpu_inputs[i], frame_bytes, cudaMemcpyHostToDevice,
                      slot.stream);
    }
  }

  {
    // Only queues the kernels; the GPU rows show when they actually run.
    ProfileRange range(Stage::Infer, "enqueue");
    slot.context->enqueueV3(slot.stream);
  }

  Dims output_dims = slot.context->getTensorShape(outputName.c_str());
  std::size_t output_elements = 1;
  for (int i = 0; i < output_dims.nbDims; i++)
    output_elements *= output_dims.d[i];
  std::size_t output_bytes = output_elements * sizeof(float);

  {
    ProfileRange range(Stage::Infer, "copy out + wait gpu");
    cudaMemcpyAsync(cpu_output, slot.outputBuffer, output_bytes,
                    cudaMemcpyDeviceToHost, slot.stream);
    cudaStreamSynchronize(slot.stream);
  }

  return true;
}

void *CudaEngine::allocate_input_frame() {
  void *frame = nullptr;
  cudaError_t err =
      cudaHostAlloc(&frame, input_frame_bytes(), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    log_msg(LOG_ERROR, "engine",
            std::string("could not allocate pinned input frame: ") +
                cudaGetErrorString(err));
    return nullptr;
  }
  return frame;
}

void CudaEngine::free_input_frame(void *frame) { cudaFreeHost(frame); }

CudaEngine::~CudaEngine() {
  log_msg(LOG_INFO, "engine", "destroying TensorRT resources");
  free_input_frames();
  for (Slot &slot : slots) {
    for (void *buf : slot.gpuBuffers)
      if (buf)
        cudaFree(buf);
    if (slot.stream)
      cudaStreamDestroy(slot.stream);
  }
  slots.clear();
}

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Inference hardware. Each device maps to one backend in libs/engine/:
//   Cuda     -> cuda.cxx     (TensorRT on NVIDIA GPUs)
//   Rocm     -> rocm.cxx     (MIGraphX on AMD GPUs)
//   Openvino -> openvino.cxx (OpenVINO on any x86 CPU or Intel GPU)
// Auto picks the first backend compiled into this build, in that order.
enum class Device { Auto, Cuda, Rocm, Openvino };

const char *device_name(Device device);
bool parse_device(const std::string &name, Device &device);
bool device_available(Device device);
Device resolve_device(Device device);

// Element type of the model input, which also fixes its layout:
//   Float32 -> planar 3x640x640 RGB scaled to 0..1 (a plain YOLO export)
//   Uint8   -> interleaved 640x640x3 RGB bytes, converted inside the model
//              (tools/add_nms.py --uint8-input)
enum class InputType { Float32, Uint8 };

class Engine;

// Returns an input frame to the engine's pool instead of freeing it.
struct InputFrameReturn {
  Engine *engine = nullptr;
  void operator()(void *frame) const;
};

// One preprocessed frame, input_frame_bytes() of host memory.
using InputFrame = std::unique_ptr<void, InputFrameReturn>;

// Common interface every backend implements. Buffers are host (CPU) memory:
// cpu_inputs[i] is frame i of the batch (from acquire_input_frame) and
// cpu_output receives batch_size * outputElementsPerFrame floats.
class Engine {
public:
  virtual ~Engine();

  // slot is 0..slotCount-1. Calls with different slots may run at the same
  // time from different threads; calls with the same slot must not.
  virtual bool infer(int slot, const void *const *cpu_inputs,
                     float *cpu_output, int batch_size) = 0;

  // A host buffer for one frame, reused from a pool and returned to it when
  // the InputFrame is destroyed. Safe to call from any thread. Backends can
  // hand out memory their device copies from directly (pinned on CUDA).
  InputFrame acquire_input_frame();
  void release_input_frame(void *frame);
  std::size_t input_frame_bytes() const;

  InputType inputType = InputType::Float32;
  std::size_t inputElementsPerFrame = 0;
  std::size_t outputElementsPerFrame = 0;
  // Output shape of one frame, batch dimension excluded. YOLO exports give
  // {300, 6} with NMS included, or {4 + classes, anchors} for the raw head.
  std::vector<std::size_t> outputDims;
  int maxBatchSize = 0;
  // How many batches the backend can run at once.
  int slotCount = 1;

protected:
  virtual void *allocate_input_frame();
  virtual void free_input_frame(void *frame);
  // Frees every pooled frame. A backend that overrides the two functions
  // above must call this in its own destructor, while its overrides still run.
  void free_input_frames();

private:
  std::mutex pool_mutex;
  std::vector<void *> free_frames;
  std::size_t allocated_frames = 0;
};

// Compile an ONNX model into a device-specific engine file.
bool build_engine_file(Device device, const char *onnx_path,
                       const char *engine_path, int batch_size);

// Load an engine file built for the same device. Returns nullptr on failure.
std::unique_ptr<Engine> load_engine(Device device, const char *engine_path,
                                    int batch_size);

// Shared by backends: create the parent directory of an output file.
bool create_parent_directory(const char *path);

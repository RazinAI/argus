#include "engine.hxx"
#include "logger.hxx"
#include <filesystem>

#ifdef ARGUS_WITH_TENSORRT
#include "cuda.hxx"
#endif
#ifdef ARGUS_WITH_MIGRAPHX
#include "rocm.hxx"
#endif
#ifdef ARGUS_WITH_OPENVINO
#include "openvino.hxx"
#endif

const char *device_name(Device device) {
  switch (device) {
  case Device::Auto:
    return "auto";
  case Device::Cuda:
    return "cuda";
  case Device::Rocm:
    return "rocm";
  case Device::Openvino:
    return "openvino";
  }
  return "unknown";
}

bool parse_device(const std::string &name, Device &device) {
  for (Device d :
       {Device::Auto, Device::Cuda, Device::Rocm, Device::Openvino}) {
    if (name == device_name(d)) {
      device = d;
      return true;
    }
  }
  return false;
}

bool device_available(Device device) {
  switch (device) {
  case Device::Auto:
    return false;
  case Device::Cuda:
#ifdef ARGUS_WITH_TENSORRT
    return true;
#else
    return false;
#endif
  case Device::Rocm:
#ifdef ARGUS_WITH_MIGRAPHX
    return true;
#else
    return false;
#endif
  case Device::Openvino:
#ifdef ARGUS_WITH_OPENVINO
    return true;
#else
    return false;
#endif
  }
  return false;
}

Device resolve_device(Device device) {
  if (device != Device::Auto)
    return device;

  for (Device d : {Device::Cuda, Device::Rocm, Device::Openvino})
    if (device_available(d))
      return d;

  return Device::Auto;
}

bool build_engine_file(Device device, const char *onnx_path,
                       const char *engine_path, int batch_size) {
  device = resolve_device(device);
  if (!device_available(device)) {
    log_msg(LOG_CRITICAL, "engine",
            std::string("backend not compiled into this build: ") +
                device_name(device));
    return false;
  }

  log_msg(LOG_INFO, "engine",
          std::string("building engine for device ") + device_name(device));

  switch (device) {
#ifdef ARGUS_WITH_TENSORRT
  case Device::Cuda:
    return cuda::build_engine_file(onnx_path, engine_path, batch_size);
#endif
#ifdef ARGUS_WITH_MIGRAPHX
  case Device::Rocm:
    return rocm::build_engine_file(onnx_path, engine_path, batch_size);
#endif
#ifdef ARGUS_WITH_OPENVINO
  case Device::Openvino:
    return openvino::build_engine_file(onnx_path, engine_path, batch_size);
#endif
  default:
    return false;
  }
}

std::unique_ptr<Engine> load_engine(Device device, const char *engine_path,
                                    int batch_size) {
  device = resolve_device(device);
  if (!device_available(device)) {
    log_msg(LOG_CRITICAL, "engine",
            std::string("backend not compiled into this build: ") +
                device_name(device));
    return nullptr;
  }

  log_msg(LOG_INFO, "engine",
          std::string("loading engine for device ") + device_name(device));

  switch (device) {
#ifdef ARGUS_WITH_TENSORRT
  case Device::Cuda:
    return cuda::load_engine(engine_path, batch_size);
#endif
#ifdef ARGUS_WITH_MIGRAPHX
  case Device::Rocm:
    return rocm::load_engine(engine_path, batch_size);
#endif
#ifdef ARGUS_WITH_OPENVINO
  case Device::Openvino:
    return openvino::load_engine(engine_path, batch_size);
#endif
  default:
    return nullptr;
  }
}

void InputFrameReturn::operator()(void *frame) const {
  if (engine)
    engine->release_input_frame(frame);
}

Engine::~Engine() { free_input_frames(); }

InputFrame Engine::acquire_input_frame() {
  {
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!free_frames.empty()) {
      void *frame = free_frames.back();
      free_frames.pop_back();
      return InputFrame(frame, InputFrameReturn{this});
    }
  }

  void *frame = allocate_input_frame();
  if (!frame)
    return InputFrame(nullptr, InputFrameReturn{this});

  std::lock_guard<std::mutex> lock(pool_mutex);
  allocated_frames++;
  return InputFrame(frame, InputFrameReturn{this});
}

void Engine::release_input_frame(void *frame) {
  std::lock_guard<std::mutex> lock(pool_mutex);
  free_frames.push_back(frame);
}

std::size_t Engine::input_frame_bytes() const {
  return inputElementsPerFrame *
         (inputType == InputType::Uint8 ? 1 : sizeof(float));
}

void *Engine::allocate_input_frame() {
  return new unsigned char[input_frame_bytes()];
}

void Engine::free_input_frame(void *frame) {
  delete[] static_cast<unsigned char *>(frame);
}

void Engine::free_input_frames() {
  std::lock_guard<std::mutex> lock(pool_mutex);
  if (allocated_frames > 0)
    log_msg(LOG_INFO, "engine",
            "input frame pool held " + std::to_string(allocated_frames) +
                " frames");
  for (void *frame : free_frames)
    free_input_frame(frame);
  free_frames.clear();
  allocated_frames = 0;
}

bool create_parent_directory(const char *path) {
  if (!path || path[0] == '\0')
    return false;

  try {
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent);
  } catch (const std::filesystem::filesystem_error &err) {
    log_msg(LOG_WARNING, "engine",
            std::string("could not create output directory: ") + err.what());
    return false;
  }

  return true;
}

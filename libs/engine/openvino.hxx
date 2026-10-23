#pragma once

#include "engine.hxx"

// OpenVINO backend: runs on any x86 CPU (Intel or AMD) and on Intel GPUs.
// OpenVINO headers stay inside openvino.cxx.
namespace openvino {

bool build_engine_file(const char *onnx_path, const char *engine_path,
                       int batch_size);
std::unique_ptr<Engine> load_engine(const char *engine_path, int batch_size);

} // namespace openvino

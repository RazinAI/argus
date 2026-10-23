#pragma once

#include "engine.hxx"

// ROCm backend: MIGraphX on AMD GPUs. MIGraphX headers stay inside rocm.cxx.
namespace rocm {

bool build_engine_file(const char *onnx_path, const char *engine_path,
                       int batch_size);
std::unique_ptr<Engine> load_engine(const char *engine_path, int batch_size);

} // namespace rocm

#pragma once

#include "engine.hxx"

// CUDA backend: TensorRT on NVIDIA GPUs. TensorRT/CUDA headers stay inside cuda.cxx so the rest
// of the program builds without them.
namespace cuda {

bool build_engine_file(const char *onnx_path, const char *engine_path,
                       int batch_size);
std::unique_ptr<Engine> load_engine(const char *engine_path, int batch_size);

} // namespace cuda

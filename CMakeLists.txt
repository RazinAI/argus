cmake_minimum_required(VERSION 3.23)

project(live-od)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Inference backends. Each one is compiled in when its SDK is found; turn one
# off to skip it even if installed (e.g. -DARGUS_ENABLE_MIGRAPHX=OFF).
option(ARGUS_ENABLE_TENSORRT "Build the CUDA TensorRT backend (NVIDIA GPUs)" ON)
option(ARGUS_ENABLE_MIGRAPHX "Build the ROCm MIGraphX backend (AMD GPUs)" ON)
option(ARGUS_ENABLE_OPENVINO "Build the OpenVINO backend (CPUs, Intel GPU/NPU)" ON)

add_executable(argus)

target_sources(argus
  PRIVATE
    src/main.cxx
    src/decoder.cxx
    src/encoder.cxx
    src/server.cxx
    src/preprocess.cxx
    src/postprocess.cxx
    libs/engine/engine.cxx
    utils/streams.cxx
    utils/logger.cxx
)

target_include_directories(argus
  PRIVATE
    utils
    libs
    libs/engine
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libswscale
    libavutil
)

target_link_libraries(argus
  PRIVATE
    PkgConfig::LIBAV
)

set(ARGUS_BACKENDS "")

if(ARGUS_ENABLE_TENSORRT)
  find_package(CUDAToolkit QUIET)
  find_library(TENSORRT_LIBRARY nvinfer)
  find_library(TENSORRT_ONNXPARSER_LIBRARY nvonnxparser)

  if(CUDAToolkit_FOUND AND TENSORRT_LIBRARY AND TENSORRT_ONNXPARSER_LIBRARY)
    target_sources(argus PRIVATE libs/engine/cuda.cxx)
    target_compile_definitions(argus PRIVATE ARGUS_WITH_TENSORRT)
    target_link_libraries(argus
      PRIVATE
        CUDA::cudart
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNXPARSER_LIBRARY}
    )
    list(APPEND ARGUS_BACKENDS "cuda (TensorRT)")
  endif()
endif()

if(ARGUS_ENABLE_MIGRAPHX)
  # ROCm installs under /opt/rocm by default.
  find_package(migraphx QUIET PATHS /opt/rocm)

  if(migraphx_FOUND)
    target_sources(argus PRIVATE libs/engine/rocm.cxx)
    target_compile_definitions(argus PRIVATE ARGUS_WITH_MIGRAPHX)
    target_link_libraries(argus PRIVATE migraphx::c)
    list(APPEND ARGUS_BACKENDS "rocm (MIGraphX)")
  endif()
endif()

if(ARGUS_ENABLE_OPENVINO)
  find_package(OpenVINO QUIET COMPONENTS Runtime)

  if(OpenVINO_FOUND)
    target_sources(argus PRIVATE libs/engine/openvino.cxx)
    target_compile_definitions(argus PRIVATE ARGUS_WITH_OPENVINO)
    target_link_libraries(argus PRIVATE openvino::runtime)
    list(APPEND ARGUS_BACKENDS "openvino (OpenVINO)")
  endif()
endif()

# NVTX labels for Nsight Systems (utils/profiling.hxx). NVTX is header-only
# and ships with the CUDA Toolkit; without it the labels compile to nothing.
find_package(CUDAToolkit QUIET)
if(TARGET CUDA::nvtx3)
  target_compile_definitions(argus PRIVATE ARGUS_WITH_NVTX)
  target_link_libraries(argus PRIVATE CUDA::nvtx3)
endif()

if(NOT ARGUS_BACKENDS)
  message(FATAL_ERROR "No inference backend found. Install TensorRT + CUDA, "
                      "ROCm MIGraphX, or OpenVINO.")
endif()

message(STATUS "argus inference backends: ${ARGUS_BACKENDS}")

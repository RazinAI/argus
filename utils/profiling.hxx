#pragma once

// NVTX labels for Nsight Systems. Record them with
//   nsys profile --trace=cuda,nvtx --cpuctxsw=process-tree ./build/argus ...
// and each labeled block shows up as a colored bar on its thread's row.
//
// NVTX calls cost almost nothing when no profiler is attached. Builds without
// the CUDA Toolkit (ARGUS_WITH_NVTX undefined) compile all of this to nothing.

#include <cstdint>

#ifdef ARGUS_WITH_NVTX
#include <cstdio>
#include <nvtx3/nvToolsExt.h>
#include <unistd.h>
#endif

// One color per pipeline stage, so a zoomed-out timeline shows which pool
// is busy.
enum class Stage { Decode, Preprocess, Infer, Postprocess, Draw, Encode };

inline const char *stage_name(Stage stage) {
  switch (stage) {
  case Stage::Decode:
    return "decode";
  case Stage::Preprocess:
    return "preprocess";
  case Stage::Infer:
    return "infer";
  case Stage::Postprocess:
    return "postprocess";
  case Stage::Draw:
    return "draw";
  case Stage::Encode:
    return "encode";
  }
  return "unknown";
}

inline std::uint32_t stage_color(Stage stage) {
  switch (stage) {
  case Stage::Decode:
    return 0xff4c9be8; // blue
  case Stage::Preprocess:
    return 0xff3fb37f; // green
  case Stage::Infer:
    return 0xffe8554c; // red
  case Stage::Postprocess:
    return 0xffe8a33c; // orange
  case Stage::Draw:
    return 0xff9b6be8; // purple
  case Stage::Encode:
    return 0xff3cc4c4; // teal
  }
  return 0xff888888;
}

// Name the calling thread's row in the timeline, e.g. "decode 2".
inline void profile_thread_name(Stage stage, int index) {
#ifdef ARGUS_WITH_NVTX
  char name[32];
  std::snprintf(name, sizeof(name), "%s %d", stage_name(stage), index);
  nvtxNameOsThreadA(static_cast<std::uint32_t>(gettid()), name);
#else
  (void)stage;
  (void)index;
#endif
}

// A labeled block on the calling thread. The range starts when the object is
// created and ends when it goes out of scope. Keep queue push/pop outside of
// it, so bars mean working and gaps mean waiting.
class ProfileRange {
public:
  ProfileRange(Stage stage, const char *message) { push(stage, message); }

  // Label for one frame, e.g. "decode s3 f120".
  ProfileRange(Stage stage, int stream_index, std::int64_t frame_id) {
#ifdef ARGUS_WITH_NVTX
    char message[64];
    std::snprintf(message, sizeof(message), "%s s%d f%lld", stage_name(stage),
                  stream_index, static_cast<long long>(frame_id));
    push(stage, message);
#else
    (void)stage;
    (void)stream_index;
    (void)frame_id;
#endif
  }

  ~ProfileRange() {
#ifdef ARGUS_WITH_NVTX
    nvtxRangePop();
#endif
  }

  ProfileRange(const ProfileRange &) = delete;
  ProfileRange &operator=(const ProfileRange &) = delete;

private:
  void push(Stage stage, const char *message) {
#ifdef ARGUS_WITH_NVTX
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = stage_color(stage);
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = message;
    nvtxRangePushEx(&attr);
#else
    (void)stage;
    (void)message;
#endif
  }
};

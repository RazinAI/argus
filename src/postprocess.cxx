#include "postprocess.hxx"
#include <algorithm>
#include <cstdint>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

namespace {

struct YuvColor {
  uint8_t y, u, v;
};

namespace colors {
constexpr YuvColor White = {235, 128, 128};
constexpr YuvColor Red = {76, 84, 255};
constexpr YuvColor Green = {149, 43, 21};
constexpr YuvColor Blue = {29, 255, 107};
constexpr YuvColor Yellow = {210, 16, 146};
constexpr YuvColor Cyan = {170, 166, 16};
constexpr YuvColor Magenta = {106, 202, 222};
constexpr YuvColor Orange = {156, 47, 200};
constexpr YuvColor Purple = {69, 192, 168};
constexpr YuvColor Pink = {180, 122, 191};
constexpr YuvColor Lime = {184, 41, 87};
constexpr YuvColor Teal = {120, 149, 84};
} // namespace colors

} // namespace

namespace {

// Ultralytics predict defaults.
constexpr float kConfThreshold = 0.25f;
constexpr float kIouThreshold = 0.7f;
constexpr int kMaxDetections = 300;

float iou(const Detection &a, const Detection &b) {
  float w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  float h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (w <= 0.0f || h <= 0.0f)
    return 0.0f;
  float inter = w * h;
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter);
}

// Boxes come out of the model in 640x640 letterboxed input space. Undo the
// letterbox padding and scale, then clamp to the source frame.
Detection to_source(Detection d, int srcW, int srcH,
                    const LetterboxTransform &letterbox) {
  d.x1 = std::clamp((d.x1 - letterbox.pad_x) / letterbox.scale, 0.0f,
                    (float)(srcW - 1));
  d.y1 = std::clamp((d.y1 - letterbox.pad_y) / letterbox.scale, 0.0f,
                    (float)(srcH - 1));
  d.x2 = std::clamp((d.x2 - letterbox.pad_x) / letterbox.scale, 0.0f,
                    (float)(srcW - 1));
  d.y2 = std::clamp((d.y2 - letterbox.pad_y) / letterbox.scale, 0.0f,
                    (float)(srcH - 1));
  return d;
}

std::vector<Detection> postprocess_nms(const float *output, int maxDets) {
  std::vector<Detection> dets;
  for (int i = 0; i < maxDets; i++) {
    const float *row = output + i * 6;
    if (row[4] <= 0.0f)
      break; // Output is padded with 0s up to maxDets
    dets.push_back({row[0], row[1], row[2], row[3], row[4], (int)row[5]});
  }
  return dets;
}

std::vector<Detection> postprocess_raw(const float *output, int numClasses,
                                       int numAnchors) {
  // Best class per anchor. Walk class rows in order so reads stay sequential.
  std::vector<float> best_score(numAnchors, 0.0f);
  std::vector<int> best_class(numAnchors, 0);
  for (int c = 0; c < numClasses; c++) {
    const float *scores = output + (std::size_t)(4 + c) * numAnchors;
    for (int a = 0; a < numAnchors; a++) {
      if (scores[a] > best_score[a]) {
        best_score[a] = scores[a];
        best_class[a] = c;
      }
    }
  }

  std::vector<Detection> candidates;
  for (int a = 0; a < numAnchors; a++) {
    if (best_score[a] <= kConfThreshold)
      continue;
    float cx = output[a];
    float cy = output[numAnchors + a];
    float w = output[2 * numAnchors + a];
    float h = output[3 * numAnchors + a];
    candidates.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                          best_score[a], best_class[a]});
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Detection &a, const Detection &b) {
              return a.conf > b.conf;
            });

  // Greedy per-class NMS: keep the highest score, drop same-class boxes that
  // overlap it too much.
  std::vector<Detection> dets;
  std::vector<bool> suppressed(candidates.size(), false);
  for (std::size_t i = 0; i < candidates.size(); i++) {
    if (suppressed[i])
      continue;
    dets.push_back(candidates[i]);
    if ((int)dets.size() == kMaxDetections)
      break;
    for (std::size_t j = i + 1; j < candidates.size(); j++) {
      if (!suppressed[j] && candidates[j].classId == candidates[i].classId &&
          iou(candidates[i], candidates[j]) > kIouThreshold)
        suppressed[j] = true;
    }
  }
  return dets;
}

} // namespace

bool output_layout(const std::vector<std::size_t> &dims, OutputLayout &layout) {
  if (dims.size() != 2)
    return false;

  if (dims[1] == 6) {
    layout.nmsIncluded = true;
    layout.maxDets = (int)dims[0];
    return true;
  }

  if (dims[0] > 4) {
    layout.nmsIncluded = false;
    layout.numClasses = (int)dims[0] - 4;
    layout.numAnchors = (int)dims[1];
    return true;
  }

  return false;
}

std::vector<Detection> postprocess(const float *output,
                                   const OutputLayout &layout, int srcW,
                                   int srcH,
                                   const LetterboxTransform &letterbox) {
  std::vector<Detection> dets =
      layout.nmsIncluded
          ? postprocess_nms(output, layout.maxDets)
          : postprocess_raw(output, layout.numClasses, layout.numAnchors);

  for (Detection &d : dets)
    d = to_source(d, srcW, srcH, letterbox);
  return dets;
}

namespace {

const YuvColor palette[] = {
    colors::White, colors::Red,    colors::Green, colors::Blue,
    colors::Yellow, colors::Cyan,  colors::Magenta, colors::Orange,
    colors::Purple, colors::Pink,  colors::Lime,    colors::Teal,
};
constexpr int kPaletteSize = sizeof(palette) / sizeof(palette[0]);

const YuvColor &colorForClass(int classId) {
  return palette[classId % kPaletteSize];
}

void setYuv420pPixel(AVFrame *frame, int x, int y, const YuvColor &c) {
  frame->data[0][y * frame->linesize[0] + x] = c.y;
  frame->data[1][(y >> 1) * frame->linesize[1] + (x >> 1)] = c.u;
  frame->data[2][(y >> 1) * frame->linesize[2] + (x >> 1)] = c.v;
}

void setNv12Pixel(AVFrame *frame, int x, int y, const YuvColor &c) {
  frame->data[0][y * frame->linesize[0] + x] = c.y;

  uint8_t *uv = frame->data[1] + (y >> 1) * frame->linesize[1] + (x & ~1);
  uv[0] = c.u;
  uv[1] = c.v;
}

template <typename SetPixel>
void drawBox(AVFrame *frame, int x1, int y1, int x2, int y2,
             const YuvColor &color, SetPixel set_pixel, int thickness = 2) {
  for (int t = 0; t < thickness; t++) {
    for (int x = x1; x <= x2; x++) {
      set_pixel(frame, x, y1 + t, color); // top
      set_pixel(frame, x, y2 - t, color); // bottom
    }
    for (int y = y1; y <= y2; y++) {
      set_pixel(frame, x1 + t, y, color); // left
      set_pixel(frame, x2 - t, y, color); // right
    }
  }
}

} // namespace

void draw_box(const std::vector<Detection> &dets, AVFrame *frame) {
  if (av_frame_make_writable(frame) < 0)
    return;

  int width = frame->width;
  int height = frame->height;
  for (const auto &d : dets) {
    int ix1 = std::clamp((int)d.x1, 0, width - 1);
    int iy1 = std::clamp((int)d.y1, 0, height - 1);
    int ix2 = std::clamp((int)d.x2, 0, width - 1);
    int iy2 = std::clamp((int)d.y2, 0, height - 1);
    if (ix2 <= ix1 || iy2 <= iy1)
      continue;

    const YuvColor &color = colorForClass(d.classId);
    if (frame->format == AV_PIX_FMT_NV12)
      drawBox(frame, ix1, iy1, ix2, iy2, color, setNv12Pixel);
    else if (frame->format == AV_PIX_FMT_YUV420P ||
             frame->format == AV_PIX_FMT_YUVJ420P)
      drawBox(frame, ix1, iy1, ix2, iy2, color, setYuv420pPixel);
  }
}

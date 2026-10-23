#pragma once

#include "preprocess.hxx"

extern "C" {
#include <libavutil/frame.h>
}

#include <vector>

struct Detection {
  float x1, y1, x2, y2;
  float conf;
  int classId;
};

// Layout of one frame of model output.
//   NMS included: [maxDets, 6], rows of x1, y1, x2, y2, conf, class.
//   Raw head:     [4 + numClasses, numAnchors], rows cx, cy, w, h, then one
//                 score row per class. NMS runs in postprocess().
struct OutputLayout {
  bool nmsIncluded = false;
  int maxDets = 0;
  int numClasses = 0;
  int numAnchors = 0;
};

bool output_layout(const std::vector<std::size_t> &dims, OutputLayout &layout);

std::vector<Detection> postprocess(const float *output,
                                   const OutputLayout &layout, int srcW,
                                   int srcH,
                                   const LetterboxTransform &letterbox);

void draw_box(const std::vector<Detection> &dets, AVFrame *frame);

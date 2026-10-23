#pragma once

extern "C" {
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include "engine.hxx"
#include <cstdint>
#include <memory>

struct LetterboxTransform {
  float scale = 1.0f;
  int pad_x = 0;
  int pad_y = 0;
  int resized_w = 640;
  int resized_h = 640;
};

struct Preprocessor {
  SwsContext *sws = nullptr;
  std::unique_ptr<uint8_t[]> rgbBuf;
  int dstW = 640;
  int dstH = 640;
  LetterboxTransform letterbox;
};

void initPreprocess(Preprocessor &p, int srcW, int srcH, int srcPixFmt);
// Writes one letterboxed 640x640 RGB frame into dst in the layout the model
// input takes (see InputType).
void preprocess(Preprocessor &p, AVFrame *src, void *dst, InputType type);
void destroyPreprocess(Preprocessor &p);

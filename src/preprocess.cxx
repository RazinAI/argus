#include "preprocess.hxx"
#include <algorithm>
#include <cmath>

extern "C" {
#include <libavutil/pixfmt.h>
}

void initPreprocess(Preprocessor &p, int srcW, int srcH, int srcPixFmt) {
  const float scale =
      std::min((float)p.dstW / (float)srcW, (float)p.dstH / (float)srcH);
  const int resized_w =
      std::max(1, std::min(p.dstW, (int)std::round(srcW * scale)));
  const int resized_h =
      std::max(1, std::min(p.dstH, (int)std::round(srcH * scale)));

  p.letterbox.scale = scale;
  p.letterbox.pad_x = (p.dstW - resized_w) / 2;
  p.letterbox.pad_y = (p.dstH - resized_h) / 2;
  p.letterbox.resized_w = resized_w;
  p.letterbox.resized_h = resized_h;

  p.sws = sws_getContext(srcW, srcH, (AVPixelFormat)srcPixFmt, resized_w,
                         resized_h, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr,
                         nullptr, nullptr);
  p.rgbBuf.reset(new uint8_t[p.dstW * p.dstH * 3]);
}

void preprocess(Preprocessor &p, AVFrame *src, void *dst, InputType type) {
  // A uint8 model takes swscale's interleaved RGB as is, so write it straight
  // into dst. A float model needs it scaled and planar, so stage it first.
  uint8_t *rgb = type == InputType::Uint8 ? static_cast<uint8_t *>(dst)
                                          : p.rgbBuf.get();
  std::fill(rgb, rgb + p.dstW * p.dstH * 3, 114);

  uint8_t *dstRGB[1] = {rgb + p.letterbox.pad_y * p.dstW * 3 +
                        p.letterbox.pad_x * 3};
  int dstStride[1] = {p.dstW * 3};
  sws_scale(p.sws, src->data, src->linesize, 0, src->height, dstRGB, dstStride);
  if (type == InputType::Uint8)
    return;

  float *out = static_cast<float *>(dst);
  const int nPixels = p.dstW * p.dstH;
  for (int k = 0; k < nPixels; k++) {
    out[k] = rgb[3 * k] / 255.0f;
    out[nPixels + k] = rgb[3 * k + 1] / 255.0f;
    out[2 * nPixels + k] = rgb[3 * k + 2] / 255.0f;
  }
}

void destroyPreprocess(Preprocessor &p) {
  if (p.sws) {
    sws_freeContext(p.sws);
    p.sws = nullptr;
  }
}

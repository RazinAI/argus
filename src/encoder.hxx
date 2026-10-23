#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
}

struct Server;

struct Encoder {
  AVCodecContext *codec_ctx = nullptr;
  AVPacket *packet = nullptr;
  int64_t next_pts = 0;
};

AVBufferRef *open_cuda_device();

bool open_encoder(Encoder &e, AVStream *in_stream,
                  AVCodecContext *decoder_codec_ctx,
                  AVBufferRef *cuda_device = nullptr);
bool open_encoder(Encoder &e, int width, int height, AVPixelFormat pix_fmt,
                  AVRational fps, AVBufferRef *cuda_device = nullptr);
bool serve_stream(Encoder &e, Server &s, AVFrame *frame);
bool flush_encoder(Encoder &e, Server &s);
void close_encoder(Encoder &e);

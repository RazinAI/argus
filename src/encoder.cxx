#include "encoder.hxx"
#include "logger.hxx"
#include "profiling.hxx"
#include "server.hxx"
#include "streams.hxx"

#include <algorithm>
#include <string>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#ifdef ARGUS_WITH_TENSORRT
#include <libavutil/hwcontext_cuda.h>
#endif
}

namespace {

enum class EncoderBackend { Nvenc, X264 };

bool supports_pixel_format(const AVCodec *codec, AVPixelFormat pix_fmt) {
  const void *configs = nullptr;
  int num_configs = 0;
  int ret = avcodec_get_supported_config(
      nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &configs, &num_configs);

  if (ret < 0)
    return false;

  if (!configs)
    return true;

  const AVPixelFormat *pix_fmts = (const AVPixelFormat *)configs;
  for (int i = 0; i < num_configs; i++) {
    if (pix_fmts[i] == pix_fmt)
      return true;
  }
  return false;
}

bool sane_fps(AVRational fps) {
  if (fps.num <= 0 || fps.den <= 0)
    return false;

  double value = av_q2d(fps);
  return value >= 1.0 && value <= 120.0;
}

AVRational stream_fps(AVStream *stream) {
  if (stream && sane_fps(stream->avg_frame_rate))
    return stream->avg_frame_rate;

  if (stream && sane_fps(stream->r_frame_rate))
    return stream->r_frame_rate;

  if (stream) {
    log_msg(LOG_WARNING, "encoder",
            "using fallback fps 30/1, avg_frame_rate=" +
                av_rational_string(stream->avg_frame_rate) +
                " r_frame_rate=" + av_rational_string(stream->r_frame_rate));
  }
  return AVRational{30, 1};
}

const char *backend_codec_name(EncoderBackend backend) {
  switch (backend) {
  case EncoderBackend::Nvenc:
    return "h264_nvenc";
  case EncoderBackend::X264:
    return "libx264";
  }
  return nullptr;
}

const char *backend_label(EncoderBackend backend) {
  switch (backend) {
  case EncoderBackend::Nvenc:
    return "NVENC";
  case EncoderBackend::X264:
    return "x264";
  }
  return "unknown";
}

void set_backend_options(AVDictionary **opts, EncoderBackend backend) {
  if (backend == EncoderBackend::Nvenc) {
    av_dict_set(opts, "preset", "p1", 0);
    av_dict_set(opts, "tune", "ull", 0);
    av_dict_set(opts, "rc", "cbr", 0);
    log_msg(LOG_INFO, "encoder", "NVENC options preset=p1 tune=ull rc=cbr");
    return;
  }

  av_dict_set(opts, "preset", "ultrafast", 0);
  av_dict_set(opts, "tune", "zerolatency", 0);
  log_msg(LOG_INFO, "encoder",
          "x264 options preset=ultrafast tune=zerolatency");
}

bool open_encoder_backend(Encoder &e, int width, int height,
                          AVPixelFormat pix_fmt, AVRational fps,
                          AVRational time_base, EncoderBackend backend,
                          AVBufferRef *cuda_device) {
  const char *codec_name = backend_codec_name(backend);
  log_msg(LOG_INFO, "encoder",
          std::string("finding H.264 encoder ") + codec_name);
  const AVCodec *codec = avcodec_find_encoder_by_name(codec_name);
  if (!codec) {
    log_msg(LOG_WARNING, "encoder",
            std::string("could not find H.264 encoder ") + codec_name);
    return false;
  }

  if (!supports_pixel_format(codec, pix_fmt)) {
    log_msg(LOG_WARNING, "encoder",
            std::string(codec_name) + " does not support input pixel format " +
                av_pixel_format_name(pix_fmt));
    return false;
  }

  log_msg(LOG_INFO, "encoder",
          std::string("allocating encoder context for ") + codec->name);
  e.codec_ctx = avcodec_alloc_context3(codec);
  if (!e.codec_ctx) {
    log_msg(LOG_ERROR, "encoder", "could not allocate encoder context");
    return false;
  }

  if (!sane_fps(fps)) {
    log_msg(LOG_WARNING, "encoder", "using fallback fps 30/1");
    fps = AVRational{30, 1};
  }

  int fps_rounded = (int)(av_q2d(fps) + 0.5);
  if (fps_rounded <= 0)
    fps_rounded = 30;

  e.codec_ctx->width = width;
  e.codec_ctx->height = height;
  e.codec_ctx->pix_fmt = pix_fmt;
  // Encode on the source's clock so frames keep their real spacing.
  e.codec_ctx->time_base = time_base;
  e.frame_duration =
      std::max<int64_t>(1, av_rescale_q(1, av_inv_q(fps), time_base));
  e.codec_ctx->framerate = fps;
  e.codec_ctx->bit_rate = 4000000;
  e.codec_ctx->gop_size = fps_rounded;
  e.codec_ctx->max_b_frames = 0;
  if (backend == EncoderBackend::X264)
    e.codec_ctx->thread_count = 1;
  if (backend == EncoderBackend::Nvenc && cuda_device) {
    e.codec_ctx->hw_device_ctx = av_buffer_ref(cuda_device);
    if (!e.codec_ctx->hw_device_ctx) {
      log_msg(LOG_ERROR, "encoder", "could not reference the CUDA device");
      close_encoder(e);
      return false;
    }
  }

  log_msg(LOG_INFO, "encoder",
          "config backend=" + std::string(backend_label(backend)) +
              " width=" + std::to_string(e.codec_ctx->width) +
              " height=" + std::to_string(e.codec_ctx->height) +
              " pix_fmt=" + av_pixel_format_name(e.codec_ctx->pix_fmt) +
              " fps=" + av_rational_string(fps) +
              " time_base=" + av_rational_string(e.codec_ctx->time_base) +
              " bitrate=" + std::to_string(e.codec_ctx->bit_rate) +
              " gop=" + std::to_string(e.codec_ctx->gop_size) +
              " max_b_frames=" + std::to_string(e.codec_ctx->max_b_frames) +
              " threads=" + std::to_string(e.codec_ctx->thread_count));

  AVDictionary *opts = nullptr;
  set_backend_options(&opts, backend);

  log_msg(LOG_INFO, "encoder",
          std::string("avcodec_open2 started for ") + codec_name);
  int ret = avcodec_open2(e.codec_ctx, codec, &opts);
  log_msg(LOG_INFO, "encoder",
          std::string("avcodec_open2 finished for ") + codec_name);
  av_dict_free(&opts);
  if (ret < 0) {
    std::string where = std::string("opening ") + codec_name + " failed";
    log_av_error(LOG_WARNING, "encoder", where.c_str(), ret);
    close_encoder(e);
    return false;
  }

  e.packet = av_packet_alloc();
  if (!e.packet) {
    log_msg(LOG_ERROR, "encoder", "could not allocate encoder packet");
    close_encoder(e);
    return false;
  }

  log_msg(LOG_INFO, "encoder", std::string("encoder opened: ") + codec->name);
  return true;
}

} // namespace

AVBufferRef *open_cuda_device() {
  AVBufferRef *device = nullptr;
  // Share TensorRT's (primary) context: work from separate contexts can't
  // overlap on the GPU, it takes turns in time slices.
  int flags = 0;
#ifdef ARGUS_WITH_TENSORRT
  flags = AV_CUDA_USE_PRIMARY_CONTEXT;
#endif
  int ret = av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, nullptr,
                                   nullptr, flags);
  if (ret < 0) {
    log_av_error(LOG_WARNING, "encoder",
                 "no shared CUDA device; each NVENC encoder makes its own",
                 ret);
    return nullptr;
  }

  log_msg(LOG_INFO, "encoder", "shared CUDA device opened for NVENC");
  return device;
}

bool open_encoder(Encoder &e, AVStream *in_stream,
                  AVCodecContext *decoder_codec_ctx,
                  AVBufferRef *cuda_device) {
  if (!decoder_codec_ctx) {
    log_msg(LOG_ERROR, "encoder", "decoder codec context is null");
    return false;
  }

  return open_encoder(e, decoder_codec_ctx->width, decoder_codec_ctx->height,
                      (AVPixelFormat)decoder_codec_ctx->pix_fmt,
                      stream_fps(in_stream), in_stream->time_base,
                      cuda_device);
}

bool open_encoder(Encoder &e, int width, int height, AVPixelFormat pix_fmt,
                  AVRational fps, AVRational time_base,
                  AVBufferRef *cuda_device) {
  if (open_encoder_backend(e, width, height, pix_fmt, fps, time_base,
                           EncoderBackend::Nvenc, cuda_device))
    return true;

  log_msg(LOG_WARNING, "encoder",
          "falling back to libx264 software encoder");
  if (open_encoder_backend(e, width, height, pix_fmt, fps, time_base,
                           EncoderBackend::X264, nullptr))
    return true;

  log_msg(LOG_ERROR, "encoder", "all H.264 encoder backends failed");
  return false;
}

bool serve_stream(Encoder &e, Server &s, AVFrame *frame) {
  if (!e.codec_ctx || !e.packet || !frame)
    return false;

  // Use the source timestamps, not a frame counter at the declared rate: a
  // camera that declares 30 fps but sends 15 would otherwise play back at
  // double speed. Gaps (skipped live segments) stay gaps.
  int64_t ts = frame->best_effort_timestamp;
  if (ts == AV_NOPTS_VALUE)
    ts = frame->pts;
  int64_t pts;
  if (ts == AV_NOPTS_VALUE) {
    pts = e.last_pts == AV_NOPTS_VALUE ? 0 : e.last_pts + e.frame_duration;
  } else {
    if (e.first_ts == AV_NOPTS_VALUE)
      e.first_ts = ts;
    pts = ts - e.first_ts;
    // A live stream can jump back (playlist discontinuity or a timestamp
    // reset); carry on one frame after the last one instead.
    if (e.last_pts != AV_NOPTS_VALUE && pts <= e.last_pts) {
      e.first_ts += pts - (e.last_pts + e.frame_duration);
      pts = e.last_pts + e.frame_duration;
    }
  }
  e.last_pts = pts;
  frame->pts = pts;

  int ret;
  {
    ProfileRange range(Stage::Encode, "send frame");
    ret = avcodec_send_frame(e.codec_ctx, frame);
  }
  if (ret < 0) {
    log_av_error(LOG_ERROR, "encoder", "sending frame to encoder failed", ret);
    return false;
  }

  while (true) {
    {
      ProfileRange range(Stage::Encode, "receive packet");
      ret = avcodec_receive_packet(e.codec_ctx, e.packet);
    }
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return true;
    if (ret < 0) {
      log_av_error(LOG_ERROR, "encoder", "receiving packet from encoder failed",
                   ret);
      return false;
    }

    ProfileRange range(Stage::Encode, "mux");
    if (!write_packet(s, e.packet, e.codec_ctx->time_base))
      return false;
  }
}

bool flush_encoder(Encoder &e, Server &s) {
  if (!e.codec_ctx || !e.packet)
    return true;

  int ret = avcodec_send_frame(e.codec_ctx, nullptr);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "encoder", "sending flush to encoder failed", ret);
    return false;
  }

  while (true) {
    ret = avcodec_receive_packet(e.codec_ctx, e.packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return true;
    if (ret < 0) {
      log_av_error(LOG_ERROR, "encoder", "flushing encoder failed", ret);
      return false;
    }

    if (!write_packet(s, e.packet, e.codec_ctx->time_base))
      return false;
  }
}

void close_encoder(Encoder &e) {
  log_msg(LOG_INFO, "encoder", "closing encoder");
  if (e.packet)
    av_packet_free(&e.packet);
  if (e.codec_ctx)
    avcodec_free_context(&e.codec_ctx);
  e.first_ts = AV_NOPTS_VALUE;
  e.last_pts = AV_NOPTS_VALUE;
  e.frame_duration = 1;
}

#include "decoder.hxx"
#include "logger.hxx"
#include "streams.hxx"
#include <string>
extern "C" {
#include <libavutil/dict.h>
}

namespace {

const char *nvidia_decoder_name(AVCodecID codec_id) {
  switch (codec_id) {
  case AV_CODEC_ID_H264:
    return "h264_cuvid";
  case AV_CODEC_ID_HEVC:
    return "hevc_cuvid";
  case AV_CODEC_ID_AV1:
    return "av1_cuvid";
  case AV_CODEC_ID_MJPEG:
    return "mjpeg_cuvid";
  case AV_CODEC_ID_MPEG1VIDEO:
    return "mpeg1_cuvid";
  case AV_CODEC_ID_MPEG2VIDEO:
    return "mpeg2_cuvid";
  case AV_CODEC_ID_MPEG4:
    return "mpeg4_cuvid";
  case AV_CODEC_ID_VP8:
    return "vp8_cuvid";
  case AV_CODEC_ID_VP9:
    return "vp9_cuvid";
  default:
    return nullptr;
  }
}

bool open_codec(Decoder &d, const AVCodec *codec,
                const AVCodecParameters *codecParams) {
  avcodec_free_context(&d.codecCtx);

  log_msg(LOG_INFO, "decoder",
          std::string("allocating decoder context for ") + codec->name);
  d.codecCtx = avcodec_alloc_context3(codec);
  if (!d.codecCtx) {
    log_msg(LOG_ERROR, "decoder", "could not allocate decoder context");
    return false;
  }

  int ret = avcodec_parameters_to_context(d.codecCtx, codecParams);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "decoder", "could not copy codec parameters", ret);
    return false;
  }

  log_msg(LOG_INFO, "decoder", "avcodec_open2 started");
  ret = avcodec_open2(d.codecCtx, codec, nullptr);
  log_msg(LOG_INFO, "decoder", "avcodec_open2 finished");
  if (ret < 0) {
    log_av_error(LOG_ERROR, "decoder", "could not open decoder", ret);
    return false;
  }

  return true;
}

} // namespace

bool openStream(Decoder &d, const char *url) {
  log_msg(LOG_INFO, "decoder",
          std::string("allocating format context for ") + url);
  d.formatCtx = avformat_alloc_context();
  if (!d.formatCtx) {
    log_msg(LOG_ERROR, "decoder", "could not allocate format context");
    return false;
  }

  StreamType type = stream_type_from_url(url);
  d.type = type;

  AVDictionary *options = nullptr;
  if (type == StreamType::Rtsp) {
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    log_msg(LOG_INFO, "decoder", "RTSP input transport: tcp");
  } else if (type == StreamType::Hls) {
    av_dict_set(&options, "rw_timeout", "10000000",
                0);                             // 10s network I/O timeout
    av_dict_set(&options, "reconnect", "1", 0); // reconnect before EOF
    av_dict_set(&options, "reconnect_streamed", "1",
                0); // reconnect live streams
    av_dict_set(&options, "reconnect_on_network_error", "1",
                0); // retry network errors
    av_dict_set(&options, "reconnect_delay_max", "5",
                0); // cap retry delay at 5s
    av_dict_set(&options, "user_agent", "Mozilla/5.0",
                0); // avoid default FFmpeg UA blocks

    av_dict_set(&options, "live_start_index", "-1", 0); // start near live edge
    av_dict_set(&options, "http_persistent", "1", 0); // reuse HTTP connections
    av_dict_set(&options, "http_multiple", "1",
                0); // allow parallel HTTP fetches
    av_dict_set(&options, "seg_max_retry", "3", 0); // retry failed HLS segments
  } else if (type == StreamType::Webrtc) {
    av_dict_set(&options, "rw_timeout", "10000000",
                0);                             // 10s network I/O timeout
    av_dict_set(&options, "reconnect", "1", 0); // reconnect before EOF
    av_dict_set(&options, "reconnect_streamed", "1",
                0); // reconnect live streams
    av_dict_set(&options, "reconnect_on_network_error", "1",
                0); // retry network errors
    av_dict_set(&options, "reconnect_delay_max", "5",
                0); // cap retry delay at 5s
    av_dict_set(&options, "user_agent", "Mozilla/5.0",
                0); // avoid default FFmpeg UA blocks

    log_msg(LOG_WARNING, "decoder",
            "WebRTC input requires FFmpeg WebRTC/WHEP demuxer support");
  } else if (type == StreamType::Http) {
    av_dict_set(&options, "rw_timeout", "10000000",
                0);                             // 10s network I/O timeout
    av_dict_set(&options, "reconnect", "1", 0); // reconnect before EOF
    av_dict_set(&options, "reconnect_streamed", "1",
                0); // reconnect live streams
    av_dict_set(&options, "reconnect_on_network_error", "1",
                0); // retry network errors
    av_dict_set(&options, "reconnect_delay_max", "5",
                0); // cap retry delay at 5s
    av_dict_set(&options, "user_agent", "Mozilla/5.0",
                0); // avoid default FFmpeg UA blocks
  }

  log_msg(LOG_INFO, "decoder", "avformat_open_input started");
  int ret = avformat_open_input(&d.formatCtx, url, nullptr, &options);
  av_dict_free(&options);
  log_msg(LOG_INFO, "decoder", "avformat_open_input finished");
  if (ret < 0) {
    log_av_error(LOG_ERROR, "decoder", "could not open input stream", ret);
    return false;
  }

  log_msg(LOG_INFO, "decoder", "avformat_find_stream_info started");
  ret = avformat_find_stream_info(d.formatCtx, nullptr);
  log_msg(LOG_INFO, "decoder", "avformat_find_stream_info finished");
  if (ret < 0) {
    log_av_error(LOG_ERROR, "decoder", "could not get stream info", ret);
    return false;
  }

  log_msg(LOG_INFO, "decoder",
          "input contains " + std::to_string(d.formatCtx->nb_streams) +
              " streams");

  const AVCodec *codec = nullptr;
  AVCodecParameters *codecParams = nullptr;
  for (unsigned i = 0; i < d.formatCtx->nb_streams; i++) {
    AVCodecParameters *p = d.formatCtx->streams[i]->codecpar;
    if (p->codec_type == AVMEDIA_TYPE_VIDEO && d.videoStreamIndex == -1) {
      d.videoStreamIndex = (int)i;
      codecParams = p;
      if (const char *name = nvidia_decoder_name(p->codec_id))
        codec = avcodec_find_decoder_by_name(name);
      if (!codec)
        codec = avcodec_find_decoder(p->codec_id);

      AVStream *stream = d.formatCtx->streams[i];
      log_msg(LOG_INFO, "decoder",
              "selected video stream index=" + std::to_string(i) +
                  " codec=" + (codec ? codec->name : "(unknown)") +
                  " codec_id=" + std::to_string(p->codec_id) +
                  " resolution=" + std::to_string(p->width) + "x" +
                  std::to_string(p->height) + " avg_frame_rate=" +
                  av_rational_string(stream->avg_frame_rate) +
                  " r_frame_rate=" + av_rational_string(stream->r_frame_rate) +
                  " time_base=" + av_rational_string(stream->time_base));
      break;
    }
  }

  if (d.videoStreamIndex == -1 || !codec) {
    log_msg(LOG_ERROR, "decoder",
            "stream does not contain a supported video stream");
    return false;
  }

  if (!open_codec(d, codec, codecParams)) {
    // CUVID needs a working NVIDIA GPU. On AMD/Intel machines (or a broken
    // NVIDIA driver) fall back to FFmpeg's software decoder.
    const AVCodec *software = avcodec_find_decoder(codecParams->codec_id);
    if (!software || software == codec)
      return false;

    log_msg(LOG_WARNING, "decoder",
            std::string("falling back to decoder ") + software->name);
    if (!open_codec(d, software, codecParams))
      return false;
  }

  log_msg(LOG_INFO, "decoder",
          "decoder opened pixel_format=" +
              av_pixel_format_name(d.codecCtx->pix_fmt) +
              " coded_size=" + std::to_string(d.codecCtx->width) + "x" +
              std::to_string(d.codecCtx->height));

  d.frame = av_frame_alloc();
  d.packet = av_packet_alloc();
  if (!d.frame || !d.packet)
    log_msg(LOG_ERROR, "decoder", "could not allocate frame or packet");
  return d.frame && d.packet;
}

bool nextFrame(Decoder &d) {
  int r = avcodec_receive_frame(d.codecCtx, d.frame);
  if (r == 0)
    return true;
  if (r != AVERROR(EAGAIN) && r != AVERROR_EOF) {
    log_av_error(LOG_ERROR, "decoder", "avcodec_receive_frame failed", r);
    return false;
  }

  while (true) {
    int read_ret = av_read_frame(d.formatCtx, d.packet);
    if (read_ret < 0) {
      if (read_ret == AVERROR_EOF)
        log_msg(LOG_WARNING, "decoder", "input stream reached EOF");
      else
        log_av_error(LOG_ERROR, "decoder", "av_read_frame failed", read_ret);
      break;
    }

    if (d.packet->stream_index == d.videoStreamIndex) {
      int s = avcodec_send_packet(d.codecCtx, d.packet);
      av_packet_unref(d.packet);
      if (s < 0) {
        log_av_error(LOG_ERROR, "decoder", "avcodec_send_packet failed", s);
        return false;
      }

      r = avcodec_receive_frame(d.codecCtx, d.frame);
      if (r == 0)
        return true;
      if (r != AVERROR(EAGAIN)) {
        log_av_error(LOG_ERROR, "decoder", "avcodec_receive_frame failed", r);
        return false;
      }
    } else {
      av_packet_unref(d.packet);
    }
  }

  int flush_ret = avcodec_send_packet(d.codecCtx, nullptr);
  if (flush_ret < 0 && flush_ret != AVERROR_EOF)
    log_av_error(LOG_WARNING, "decoder", "decoder flush send failed",
                 flush_ret);

  r = avcodec_receive_frame(d.codecCtx, d.frame);
  if (r == 0)
    return true;
  if (r != AVERROR_EOF && r != AVERROR(EAGAIN))
    log_av_error(LOG_ERROR, "decoder", "decoder flush receive failed", r);
  return false;
}

void closeStream(Decoder &d) {
  log_msg(LOG_INFO, "decoder", "closing input stream");
  if (d.formatCtx)
    avformat_close_input(&d.formatCtx);
  if (d.packet)
    av_packet_free(&d.packet);
  if (d.frame)
    av_frame_free(&d.frame);
  if (d.codecCtx)
    avcodec_free_context(&d.codecCtx);
}

#include "frame_quantize.h"
#include <string.h>

void quantize_rgb_frame_inplace(const uint8_t *src_raw,
                               int8_t *dst_q,
                               size_t bytes_per_frame)
{
    if (src_raw == nullptr || dst_q == nullptr) {
        return;
    }
    for (size_t i = 0; i < bytes_per_frame; ++i) {
        int16_t q = (int16_t)src_raw[i] - 128;
        if (q > 127) {
            q = 127;
        }
        if (q < -128) {
            q = -128;
        }
        dst_q[i] = (int8_t)q;
    }
}

void compute_checksum_from_q(const int8_t *buf_q, size_t len, ob_checksum_stats_t *stats)
{
    if (buf_q == nullptr || stats == nullptr || len == 0U) {
        return;
    }

    uint32_t sum = 0U;
    uint8_t min_v = 0xFFU;
    uint8_t max_v = 0U;
    for (size_t i = 0; i < len; ++i) {
        const int16_t restored = (int16_t)buf_q[i] + 128;
        const uint8_t v = (restored < 0) ? 0U : (restored > 255 ? 255U : (uint8_t)restored);
        sum += v;
        if (v < min_v) {
            min_v = v;
        }
        if (v > max_v) {
            max_v = v;
        }
    }
    stats->sum = sum;
    stats->min = min_v;
    stats->max = max_v;
}

void interleave_prev_curr_nhwc(int8_t *dst_6ch,
                               const int8_t *prev_q,
                               const int8_t *curr_q,
                               size_t pix_cnt)
{
    if (dst_6ch == nullptr || prev_q == nullptr || curr_q == nullptr) {
        return;
    }
    for (size_t i = 0; i < pix_cnt; ++i) {
        const size_t s3 = i * 3U;
        const size_t d6 = i * 6U;
        dst_6ch[d6 + 0] = prev_q[s3 + 0];
        dst_6ch[d6 + 1] = prev_q[s3 + 1];
        dst_6ch[d6 + 2] = prev_q[s3 + 2];
        dst_6ch[d6 + 3] = curr_q[s3 + 0];
        dst_6ch[d6 + 4] = curr_q[s3 + 1];
        dst_6ch[d6 + 5] = curr_q[s3 + 2];
    }
}

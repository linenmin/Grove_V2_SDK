#ifndef PREPROCESS_FRAME_QUANTIZE_H
#define PREPROCESS_FRAME_QUANTIZE_H

#include <stdint.h>
#include <stddef.h>
#include "ob_debug_stats.h"

#ifdef __cplusplus
extern "C" {
#endif

void quantize_rgb_frame_inplace(const uint8_t *src_raw,
                               int8_t *dst_q,
                               size_t bytes_per_frame);

void compute_checksum_from_q(const int8_t *buf_q, 
                            size_t len, 
                            ob_checksum_stats_t *stats);

void interleave_prev_curr_nhwc(int8_t *dst_6ch,
                               const int8_t *prev_q,
                               const int8_t *curr_q,
                               size_t pix_cnt);

#ifdef __cplusplus
}
#endif

#endif // PREPROCESS_FRAME_QUANTIZE_H

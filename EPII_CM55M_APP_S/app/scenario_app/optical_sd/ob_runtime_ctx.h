#ifndef OPTICAL_SD_OB_RUNTIME_CTX_H_
#define OPTICAL_SD_OB_RUNTIME_CTX_H_

#include <stdint.h>

#include "ob_debug_stats.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *raw_buf_1;
    uint8_t *raw_buf_2;

    int raw_frame_idx;
    int raw_frame_max;
    int loop_cnt;
    int log_print_interval;

    ob_checksum_stats_t raw1_stats;
    ob_checksum_stats_t raw2_stats;

    uint32_t sd_us;
    uint32_t preproc_us;
    uint32_t infer_us;
    uint32_t total_us;
} ob_runtime_ctx_t;

#ifdef __cplusplus
}
#endif

#endif  // OPTICAL_SD_OB_RUNTIME_CTX_H_

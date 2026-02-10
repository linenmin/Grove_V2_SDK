#ifndef OPTICAL_SD_OB_DEBUG_STATS_H_
#define OPTICAL_SD_OB_DEBUG_STATS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t sum;
    uint8_t min;
    uint8_t max;
} ob_checksum_stats_t;

typedef struct {
    int dx_milli;
    int dy_milli;
    int mean_dx_milli;
    int mean_dy_milli;
    int8_t out0_min;
    int8_t out0_max;
    int8_t out1_min;
    int8_t out1_max;
} ob_flow_summary_t;

void ob_compute_checksum(const uint8_t *buf, size_t len, ob_checksum_stats_t *stats);

void ob_compute_flow_summary(const int8_t *out_data,
                             int out_w,
                             int out_h,
                             int out_zp,
                             float out_scale,
                             ob_flow_summary_t *summary);

bool ob_should_log(int loop_cnt, int interval);

void ob_log_infer_line(int loop_cnt,
                       int frame_idx,
                       int frame_max,
                       const ob_flow_summary_t *summary,
                       const ob_checksum_stats_t *in1,
                       const ob_checksum_stats_t *in2,
                       uint32_t sd_us,
                       uint32_t preproc_us,
                       uint32_t infer_us,
                       uint32_t total_us);

#ifdef __cplusplus
}
#endif

#endif  // OPTICAL_SD_OB_DEBUG_STATS_H_

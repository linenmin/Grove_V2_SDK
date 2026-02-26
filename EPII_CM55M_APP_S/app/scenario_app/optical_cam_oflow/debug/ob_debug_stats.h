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
                             int out_c,
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


/** plan-009 R7: 输出整帧 mag 统计与 3x3 网格点 dx/dy，区分常量场与渲染映射问题 */
void ob_log_mag_stats_grid_sample(const int8_t *out_data,
                                  int out_w,
                                  int out_h,
                                  int out_c,
                                  int out_zp,
                                  float out_scale);

/** plan-009 R9: 输出 int8 通道分布（主峰值与边界饱和占比） */
void ob_log_out_q_histogram(const int8_t *out_data,
                            int out_w,
                            int out_h,
                            int out_c);

#ifdef __cplusplus
}
#endif

#endif  // OPTICAL_SD_OB_DEBUG_STATS_H_

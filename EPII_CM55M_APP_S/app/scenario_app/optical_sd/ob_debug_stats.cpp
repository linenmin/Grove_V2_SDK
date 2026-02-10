#include "ob_debug_stats.h"

#include "xprintf.h"

void ob_compute_checksum(const uint8_t *buf, size_t len, ob_checksum_stats_t *stats)
{
    if (buf == nullptr || stats == nullptr || len == 0U) {
        return;
    }

    uint32_t sum = 0U;
    uint8_t min_v = 0xFFU;
    uint8_t max_v = 0U;
    for (size_t i = 0; i < len; ++i) {
        const uint8_t v = buf[i];
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

void ob_compute_flow_summary(const int8_t *out_data,
                             int out_w,
                             int out_h,
                             int out_zp,
                             float out_scale,
                             ob_flow_summary_t *summary)
{
    if (out_data == nullptr || summary == nullptr || out_w <= 0 || out_h <= 0) {
        return;
    }

    const int pixels = out_w * out_h;
    const int center_idx = (out_h / 2) * out_w + (out_w / 2);

    const float dx_f = ((float)out_data[center_idx * 4 + 0] - out_zp) * out_scale;
    const float dy_f = ((float)out_data[center_idx * 4 + 1] - out_zp) * out_scale;

    int64_t sum_dx = 0;
    int64_t sum_dy = 0;
    int8_t out0_min = 127;
    int8_t out0_max = -128;
    int8_t out1_min = 127;
    int8_t out1_max = -128;

    for (int i = 0; i < pixels; ++i) {
        const int8_t v0 = out_data[i * 4 + 0];
        const int8_t v1 = out_data[i * 4 + 1];

        sum_dx += (int)v0 - out_zp;
        sum_dy += (int)v1 - out_zp;

        if (v0 < out0_min) {
            out0_min = v0;
        }
        if (v0 > out0_max) {
            out0_max = v0;
        }
        if (v1 < out1_min) {
            out1_min = v1;
        }
        if (v1 > out1_max) {
            out1_max = v1;
        }
    }

    const float mean_dx = ((float)sum_dx / (float)pixels) * out_scale;
    const float mean_dy = ((float)sum_dy / (float)pixels) * out_scale;

    summary->dx_milli = (int)(dx_f * 1000.0f);
    summary->dy_milli = (int)(dy_f * 1000.0f);
    summary->mean_dx_milli = (int)(mean_dx * 1000.0f);
    summary->mean_dy_milli = (int)(mean_dy * 1000.0f);
    summary->out0_min = out0_min;
    summary->out0_max = out0_max;
    summary->out1_min = out1_min;
    summary->out1_max = out1_max;
}

bool ob_should_log(int loop_cnt, int interval)
{
    if (interval <= 0) {
        return true;
    }
    return (loop_cnt % interval) == 0;
}

void ob_log_infer_line(int loop_cnt,
                       int frame_idx,
                       int frame_max,
                       const ob_flow_summary_t *summary,
                       const ob_checksum_stats_t *in1,
                       const ob_checksum_stats_t *in2,
                       uint32_t sd_us,
                       uint32_t preproc_us,
                       uint32_t infer_us,
                       uint32_t total_us)
{
    if (summary == nullptr || in1 == nullptr || in2 == nullptr) {
        return;
    }

    xprintf("[loop=%d][frame=%d/%d] center dx=%d.%03d dy=%d.%03d | mean dx=%d.%03d dy=%d.%03d | in1 sum=%u min=%u max=%u | in2 sum=%u min=%u max=%u | out0 min=%d max=%d out1 min=%d max=%d | times: sd=%u.%03ums preproc=%u.%03ums infer=%u.%03ums total=%u.%03ums\\r\\n",
            loop_cnt,
            frame_idx,
            frame_max,
            summary->dx_milli / 1000,
            (summary->dx_milli >= 0 ? summary->dx_milli : -summary->dx_milli) % 1000,
            summary->dy_milli / 1000,
            (summary->dy_milli >= 0 ? summary->dy_milli : -summary->dy_milli) % 1000,
            summary->mean_dx_milli / 1000,
            (summary->mean_dx_milli >= 0 ? summary->mean_dx_milli : -summary->mean_dx_milli) % 1000,
            summary->mean_dy_milli / 1000,
            (summary->mean_dy_milli >= 0 ? summary->mean_dy_milli : -summary->mean_dy_milli) % 1000,
            in1->sum,
            in1->min,
            in1->max,
            in2->sum,
            in2->min,
            in2->max,
            summary->out0_min,
            summary->out0_max,
            summary->out1_min,
            summary->out1_max,
            sd_us / 1000,
            sd_us % 1000,
            preproc_us / 1000,
            preproc_us % 1000,
            infer_us / 1000,
            infer_us % 1000,
            total_us / 1000,
            total_us % 1000);
}

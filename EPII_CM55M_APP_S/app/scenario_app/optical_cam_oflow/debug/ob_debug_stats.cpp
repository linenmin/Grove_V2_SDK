#include "ob_debug_stats.h"

#include "xprintf.h"
#include <math.h>

static int to_tenth_percent(int count, int total)
{
    if (total <= 0) {
        return 0;
    }
    return (count * 1000) / total;
}

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
                             int out_c,
                             int out_zp,
                             float out_scale,
                             ob_flow_summary_t *summary)
{
    if (out_data == nullptr || summary == nullptr || out_w <= 0 || out_h <= 0 || out_c < 2) {
        return;
    }

    const int pixels = out_w * out_h;
    const int center_idx = (out_h / 2) * out_w + (out_w / 2);
    const int stride = out_c;

    const float dx_f = ((float)out_data[center_idx * stride + 0] - out_zp) * out_scale;
    const float dy_f = ((float)out_data[center_idx * stride + 1] - out_zp) * out_scale;

    int64_t sum_dx = 0;
    int64_t sum_dy = 0;
    int8_t out0_min = 127;
    int8_t out0_max = -128;
    int8_t out1_min = 127;
    int8_t out1_max = -128;

    for (int i = 0; i < pixels; ++i) {
        const int8_t v0 = out_data[i * stride + 0];
        const int8_t v1 = out_data[i * stride + 1];

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

    xprintf("[loop=%d][frame=%d/%d] center dx=%d.%03d dy=%d.%03d | mean dx=%d.%03d dy=%d.%03d | in1 sum=%u min=%u max=%u | in2 sum=%u min=%u max=%u | out0 min=%d max=%d out1 min=%d max=%d | times: sd=%u.%03ums preproc=%u.%03ums infer=%u.%03ums total=%u.%03ums\r\n",
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

void ob_log_col_mean_mag_sample(const int8_t *out_data,
                                int out_w,
                                int out_h,
                                int out_c,
                                int out_zp,
                                float out_scale,
                                int sample_step)
{
    if (out_data == nullptr || out_w <= 0 || out_h <= 0 || out_c < 2 || sample_step <= 0) {
        return;
    }

    const int stride = out_c;
    /* plan-007: 改为密集采样连续列，消除混叠掩盖。输出前 16 列 */
    const int n_cols = (out_w > 16) ? 16 : out_w;

    xprintf("[col_mean_mag] step=1");
    for (int c = 0; c < n_cols; ++c) {
        double sum_mag = 0.0;
        for (int r = 0; r < out_h; ++r) {
            const int idx = (r * out_w + c) * stride;
            const float dx = ((float)out_data[idx + 0] - (float)out_zp) * out_scale;
            const float dy = ((float)out_data[idx + 1] - (float)out_zp) * out_scale;
            sum_mag += (double)sqrtf(dx * dx + dy * dy);
        }
        const int mean_mag_int = (int)((sum_mag / (double)out_h) * 1000.0);
        xprintf(" c%d=%d", c, mean_mag_int);
    }
    xprintf(" |");
    /* 附加中间区域的几列，帮助确认 */
    const int mid_c = out_w / 2;
    const int max_mid = (mid_c + 8 > out_w) ? out_w : mid_c + 8;
    for (int c = mid_c; c < max_mid; ++c) {
        double sum_mag = 0.0;
        for (int r = 0; r < out_h; ++r) {
            const int idx = (r * out_w + c) * stride;
            const float dx = ((float)out_data[idx + 0] - (float)out_zp) * out_scale;
            const float dy = ((float)out_data[idx + 1] - (float)out_zp) * out_scale;
            sum_mag += (double)sqrtf(dx * dx + dy * dy);
        }
        const int mean_mag_int = (int)((sum_mag / (double)out_h) * 1000.0);
        xprintf(" c%d=%d", c, mean_mag_int);
    }

    xprintf("\r\n");
}

void ob_log_mag_stats_grid_sample(const int8_t *out_data,
                                  int out_w,
                                  int out_h,
                                  int out_c,
                                  int out_zp,
                                  float out_scale)
{
    if (out_data == nullptr || out_w <= 0 || out_h <= 0 || out_c < 2) {
        return;
    }

    const int pixels = out_w * out_h;
    const int stride = out_c;

    float min_mag = 1e30f;
    float max_mag = 0.0f;
    double sum_mag = 0.0;
    double sum_mag2 = 0.0;

    for (int i = 0; i < pixels; ++i) {
        const float dx = ((float)out_data[i * stride + 0] - (float)out_zp) * out_scale;
        const float dy = ((float)out_data[i * stride + 1] - (float)out_zp) * out_scale;
        const float mag = sqrtf(dx * dx + dy * dy);
        if (mag < min_mag) {
            min_mag = mag;
        }
        if (mag > max_mag) {
            max_mag = mag;
        }
        sum_mag += (double)mag;
        sum_mag2 += (double)mag * (double)mag;
    }

    const float mean_mag = (float)(sum_mag / (double)pixels);
    float var_mag = (float)(sum_mag2 / (double)pixels) - (mean_mag * mean_mag);
    if (var_mag < 0.0f) {
        var_mag = 0.0f;
    }
    const float std_mag = sqrtf(var_mag);

    const int x_list[3] = {0, out_w / 2, out_w - 1};
    const int y_list[3] = {0, out_h / 2, out_h - 1};

    xprintf("[mag_stats] min=%d.%03d max=%d.%03d mean=%d.%03d std=%d.%03d | grid",
            (int)min_mag,
            (int)fabsf(min_mag * 1000.0f) % 1000,
            (int)max_mag,
            (int)fabsf(max_mag * 1000.0f) % 1000,
            (int)mean_mag,
            (int)fabsf(mean_mag * 1000.0f) % 1000,
            (int)std_mag,
            (int)fabsf(std_mag * 1000.0f) % 1000);

    for (int yi = 0; yi < 3; ++yi) {
        for (int xi = 0; xi < 3; ++xi) {
            const int x = x_list[xi];
            const int y = y_list[yi];
            const int idx = (y * out_w + x) * stride;
            const float dx = ((float)out_data[idx + 0] - (float)out_zp) * out_scale;
            const float dy = ((float)out_data[idx + 1] - (float)out_zp) * out_scale;
            xprintf(" (%d,%d):dx=%d.%03d,dy=%d.%03d",
                    x,
                    y,
                    (int)dx,
                    (int)fabsf(dx * 1000.0f) % 1000,
                    (int)dy,
                    (int)fabsf(dy * 1000.0f) % 1000);
        }
    }

    xprintf("\r\n");
}

void ob_log_out_q_histogram(const int8_t *out_data,
                            int out_w,
                            int out_h,
                            int out_c)
{
    if (out_data == nullptr || out_w <= 0 || out_h <= 0 || out_c < 2) {
        return;
    }

    const int pixels = out_w * out_h;
    const int stride = out_c;
    int hist0[256] = {0};
    int hist1[256] = {0};

    for (int i = 0; i < pixels; ++i) {
        const int idx = i * stride;
        const int q0 = (int)out_data[idx + 0];
        const int q1 = (int)out_data[idx + 1];
        hist0[q0 + 128]++;
        hist1[q1 + 128]++;
    }

    int top0_q = -128, top0_cnt = -1;
    int top1_q = -128, top1_cnt = -1;
    int second0_q = -128, second0_cnt = -1;
    int second1_q = -128, second1_cnt = -1;
    for (int b = 0; b < 256; ++b) {
        const int q = b - 128;
        const int c0 = hist0[b];
        const int c1 = hist1[b];

        if (c0 > top0_cnt) {
            second0_cnt = top0_cnt;
            second0_q = top0_q;
            top0_cnt = c0;
            top0_q = q;
        } else if (c0 > second0_cnt) {
            second0_cnt = c0;
            second0_q = q;
        }

        if (c1 > top1_cnt) {
            second1_cnt = top1_cnt;
            second1_q = top1_q;
            top1_cnt = c1;
            top1_q = q;
        } else if (c1 > second1_cnt) {
            second1_cnt = c1;
            second1_q = q;
        }
    }

    int near_min0 = 0;
    int near_max0 = 0;
    int near_min1 = 0;
    int near_max1 = 0;
    for (int q = -128; q <= -124; ++q) {
        near_min0 += hist0[q + 128];
        near_min1 += hist1[q + 128];
    }
    for (int q = 124; q <= 127; ++q) {
        near_max0 += hist0[q + 128];
        near_max1 += hist1[q + 128];
    }

    const int top0_pct10 = to_tenth_percent(top0_cnt, pixels);
    const int top1_pct10 = to_tenth_percent(top1_cnt, pixels);
    const int sec0_pct10 = to_tenth_percent(second0_cnt, pixels);
    const int sec1_pct10 = to_tenth_percent(second1_cnt, pixels);
    const int nmin0_pct10 = to_tenth_percent(near_min0, pixels);
    const int nmin1_pct10 = to_tenth_percent(near_min1, pixels);
    const int nmax0_pct10 = to_tenth_percent(near_max0, pixels);
    const int nmax1_pct10 = to_tenth_percent(near_max1, pixels);

    xprintf("[out_hist] ch0 top=%d(%d.%01d%%) second=%d(%d.%01d%%) near_min=%d(%d.%01d%%) near_max=%d(%d.%01d%%) | ch1 top=%d(%d.%01d%%) second=%d(%d.%01d%%) near_min=%d(%d.%01d%%) near_max=%d(%d.%01d%%)\r\n",
            top0_q,
            top0_pct10 / 10,
            top0_pct10 % 10,
            second0_q,
            sec0_pct10 / 10,
            sec0_pct10 % 10,
            near_min0,
            nmin0_pct10 / 10,
            nmin0_pct10 % 10,
            near_max0,
            nmax0_pct10 / 10,
            nmax0_pct10 % 10,
            top1_q,
            top1_pct10 / 10,
            top1_pct10 % 10,
            second1_q,
            sec1_pct10 / 10,
            sec1_pct10 % 10,
            near_min1,
            nmin1_pct10 / 10,
            nmin1_pct10 % 10,
            near_max1,
            nmax1_pct10 / 10,
            nmax1_pct10 % 10);
}

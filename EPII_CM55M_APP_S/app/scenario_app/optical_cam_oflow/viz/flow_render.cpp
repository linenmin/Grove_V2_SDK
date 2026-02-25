#include "flow_render.h"
#include "JPEGENC.h"
#include <math.h>
#include <stdlib.h>

// 1=固定 scale，0=动态映射（max + 百分位拉伸）
#ifndef FLOW_VIZ_FIXED_SCALE
#define FLOW_VIZ_FIXED_SCALE 0
#endif

// 1=先减去整帧平均 flow（全局平移）再做可视化，突出局部差异
#ifndef FLOW_VIZ_REMOVE_GLOBAL_MOTION
#define FLOW_VIZ_REMOVE_GLOBAL_MOTION 0
#endif

// 1=生成渐变测试图以检查后续 JPEG/Web 链路线条，0=正常渲染
#ifndef FLOW_VIZ_TEST_PATTERN
#define FLOW_VIZ_TEST_PATTERN 0
#endif

// 动态映射时使用百分位拉伸（当前关闭，使用固定缩放）
#ifndef FLOW_VIZ_PERCENTILE_LOW
#define FLOW_VIZ_PERCENTILE_LOW 5
#endif

#ifndef FLOW_VIZ_PERCENTILE_HIGH
#define FLOW_VIZ_PERCENTILE_HIGH 95
#endif

// 轻量空间平滑，减轻零散亮点和锯齿纹理。
#ifndef FLOW_VIZ_LIGHT_SMOOTH
#define FLOW_VIZ_LIGHT_SMOOTH 0
#endif

// 1=按行去除幅值基线（抑制整行条带），0=不做行偏置抑制
#ifndef FLOW_VIZ_REMOVE_ROW_BIAS
#define FLOW_VIZ_REMOVE_ROW_BIAS 0
#endif

// 绝对真实内存布局：纯 Planar 排布 (NCHW)
#ifndef FLOW_VIZ_OUT_PLANAR
#define FLOW_VIZ_OUT_PLANAR 0
#endif

// 灰度渲染分量模式。0: |flow| 幅值；1: dx 有符号；2: dy 有符号
#ifndef FLOW_VIZ_GRAY_COMPONENT
#define FLOW_VIZ_GRAY_COMPONENT 0
#endif

static inline void read_flow_dxdy(const int8_t *flow_data,
                                  int out_stride,
                                  int out_zp,
                                  float out_scale,
                                  int pixels,
                                  int i,
                                  float *dx,
                                  float *dy)
{
#if FLOW_VIZ_OUT_PLANAR
    const int qx = (int)flow_data[i];
    const int qy = (int)flow_data[pixels + i];
#else
    const int qx = (int)flow_data[i * out_stride + 0];
    const int qy = (int)flow_data[i * out_stride + 1];
#endif
    *dx = ((float)qx - (float)out_zp) * out_scale;
    *dy = ((float)qy - (float)out_zp) * out_scale;
}

void flow_render_to_gray(uint8_t *out_gray,
                        const int8_t *flow_data,
                        int out_w,
                        int out_h,
                        int out_stride,
                        int out_zp,
                        float out_scale)
{
    if (out_gray == nullptr || flow_data == nullptr || out_w <= 0 || out_h <= 0 || out_stride < 2) {
        return;
    }

    const int pixels = out_w * out_h;

    float mean_dx = 0.0f;
    float mean_dy = 0.0f;
#if FLOW_VIZ_REMOVE_GLOBAL_MOTION
    {
        double sum_dx = 0.0;
        double sum_dy = 0.0;
        for (int i = 0; i < pixels; ++i) {
            float dx = 0.0f;
            float dy = 0.0f;
            read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);
            sum_dx += (double)dx;
            sum_dy += (double)dy;
        }
        mean_dx = (float)(sum_dx / (double)pixels);
        mean_dy = (float)(sum_dy / (double)pixels);
    }
#endif

#if FLOW_VIZ_TEST_PATTERN
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            // 平滑渐变测试模式，便于检测垂直条纹问题是否源自后续渲染管线
            out_gray[y * out_w + x] = (uint8_t)(x % 256);
        }
    }
    return;
#endif

#if (FLOW_VIZ_GRAY_COMPONENT == 1) || (FLOW_VIZ_GRAY_COMPONENT == 2)
    // 直接可视化单分量，便于判定“空间结构是否真实存在”。
    float max_abs = 1e-6f;
    for (int i = 0; i < pixels; ++i) {
        float dx = 0.0f;
        float dy = 0.0f;
        read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);
#if FLOW_VIZ_REMOVE_GLOBAL_MOTION
        dx -= mean_dx;
        dy -= mean_dy;
#endif
        const float comp = (FLOW_VIZ_GRAY_COMPONENT == 1) ? dx : dy;
        const float a = fabsf(comp);
        if (a > max_abs) {
            max_abs = a;
        }
    }

    const float inv_max = (max_abs > 1e-6f) ? (1.0f / max_abs) : 0.0f;
    for (int i = 0; i < pixels; ++i) {
        float dx = 0.0f;
        float dy = 0.0f;
        read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);
#if FLOW_VIZ_REMOVE_GLOBAL_MOTION
        dx -= mean_dx;
        dy -= mean_dy;
#endif
        float n = ((FLOW_VIZ_GRAY_COMPONENT == 1) ? dx : dy) * inv_max; // [-1, 1]
        if (n < -1.0f) {
            n = -1.0f;
        } else if (n > 1.0f) {
            n = 1.0f;
        }
        out_gray[i] = (uint8_t)(127.5f + n * 127.0f);
    }
    return;
#endif

#if FLOW_VIZ_FIXED_SCALE
    /* R5: 降低固定缩放，先解除全白饱和，再观察是否仍近似常量场 */
    const float kFixedScale = 5.0f;
    for (int i = 0; i < pixels; ++i) {
        float dx = 0.0f;
        float dy = 0.0f;
        read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);
#if FLOW_VIZ_REMOVE_GLOBAL_MOTION
        dx -= mean_dx;
        dy -= mean_dy;
#endif
        const float mag = sqrtf(dx * dx + dy * dy);
        float v = mag * kFixedScale;
        if (v > 255.0f) {
            v = 255.0f;
        }
        out_gray[i] = (uint8_t)(v);
    }
#else
// 小尺寸缓存，避免动态分配；当前流图高度为 160。
#if FLOW_VIZ_REMOVE_ROW_BIAS
    const bool use_row_bias = (out_h <= 256);
    float row_mag_base[256];
    if (use_row_bias) {
        for (int y = 0; y < out_h; ++y) {
            double sum_mag = 0.0;
            const int row_base = y * out_w;
            for (int x = 0; x < out_w; ++x) {
                const int i = row_base + x;
                float dx = 0.0f;
                float dy = 0.0f;
                read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);
#if FLOW_VIZ_REMOVE_GLOBAL_MOTION
                dx -= mean_dx;
                dy -= mean_dy;
#endif
                const float mag = sqrtf(dx * dx + dy * dy);
                sum_mag += (double)mag;
            }
            row_mag_base[y] = (float)(sum_mag / (double)out_w);
        }
    }
#endif

    float max_mag = 1e-6f;
    for (int y = 0; y < out_h; ++y) {
#if FLOW_VIZ_REMOVE_ROW_BIAS
        const float row_base_mag = use_row_bias ? row_mag_base[y] : 0.0f;
#endif
        const int row_base = y * out_w;
        for (int x = 0; x < out_w; ++x) {
            const int i = row_base + x;
            float dx = 0.0f;
            float dy = 0.0f;
            read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);
#if FLOW_VIZ_REMOVE_GLOBAL_MOTION
            dx -= mean_dx;
            dy -= mean_dy;
#endif
            float mag = sqrtf(dx * dx + dy * dy);
#if FLOW_VIZ_REMOVE_ROW_BIAS
            if (use_row_bias) {
                mag = fabsf(mag - row_base_mag);
            }
#endif
            if (mag > max_mag) {
                max_mag = mag;
            }
        }
    }

    uint32_t hist[256] = {0};
    for (int y = 0; y < out_h; ++y) {
#if FLOW_VIZ_REMOVE_ROW_BIAS
        const float row_base_mag = use_row_bias ? row_mag_base[y] : 0.0f;
#endif
        const int row_base = y * out_w;
        for (int x = 0; x < out_w; ++x) {
            const int i = row_base + x;
            float dx = 0.0f;
            float dy = 0.0f;
            read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);
#if FLOW_VIZ_REMOVE_GLOBAL_MOTION
            dx -= mean_dx;
            dy -= mean_dy;
#endif
            float mag = sqrtf(dx * dx + dy * dy);
#if FLOW_VIZ_REMOVE_ROW_BIAS
            if (use_row_bias) {
                mag = fabsf(mag - row_base_mag);
            }
#endif
            float n = (max_mag > 1e-6f) ? (mag / max_mag) : 0.0f;
            if (n < 0.0f) {
                n = 0.0f;
            } else if (n > 1.0f) {
                n = 1.0f;
            }
            int b = (int)(n * 255.0f + 0.5f);
            if (b < 0) {
                b = 0;
            } else if (b > 255) {
                b = 255;
            }
            hist[b]++;
        }
    }

    uint32_t low_target = (uint32_t)(((uint64_t)pixels * (uint64_t)FLOW_VIZ_PERCENTILE_LOW) / 100ULL);
    uint32_t high_target = (uint32_t)(((uint64_t)pixels * (uint64_t)FLOW_VIZ_PERCENTILE_HIGH) / 100ULL);
    if (high_target <= low_target) {
        high_target = low_target + 1U;
    }

    uint32_t acc = 0U;
    int low_bin = 0;
    int high_bin = 255;
    for (int b = 0; b < 256; ++b) {
        acc += hist[b];
        if (acc > low_target) {
            low_bin = b;
            break;
        }
    }

    acc = 0U;
    for (int b = 0; b < 256; ++b) {
        acc += hist[b];
        if (acc >= high_target) {
            high_bin = b;
            break;
        }
    }

    float low_n = (float)low_bin / 255.0f;
    float high_n = (float)high_bin / 255.0f;
    if (high_n <= low_n + 1e-3f) {
        high_n = low_n + 1e-3f;
    }
    if (high_n > 1.0f) {
        high_n = 1.0f;
    }
    const float inv_span = 1.0f / (high_n - low_n);

#if FLOW_VIZ_LIGHT_SMOOTH
    uint8_t prev_row[640];
    const int smooth_w = (out_w <= 640) ? out_w : 0;
    for (int x = 0; x < smooth_w; ++x) {
        prev_row[x] = 0U;
    }

    for (int y = 0; y < out_h; ++y) {
#if FLOW_VIZ_REMOVE_ROW_BIAS
        const float row_base_mag = use_row_bias ? row_mag_base[y] : 0.0f;
#endif
        uint8_t prev_h = 0U;
        const int row_base = y * out_w;
        for (int x = 0; x < out_w; ++x) {
            const int i = row_base + x;
            float dx = 0.0f;
            float dy = 0.0f;
            read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);
#if FLOW_VIZ_REMOVE_GLOBAL_MOTION
            dx -= mean_dx;
            dy -= mean_dy;
#endif
            float mag = sqrtf(dx * dx + dy * dy);
#if FLOW_VIZ_REMOVE_ROW_BIAS
            if (use_row_bias) {
                mag = fabsf(mag - row_base_mag);
            }
#endif
            float n = (max_mag > 1e-6f) ? (mag / max_mag) : 0.0f;
            if (n < 0.0f) {
                n = 0.0f;
            } else if (n > 1.0f) {
                n = 1.0f;
            }
            float s = (n - low_n) * inv_span;
            if (s < 0.0f) {
                s = 0.0f;
            } else if (s > 1.0f) {
                s = 1.0f;
            }
            const uint8_t v = (uint8_t)(s * 255.0f + 0.5f);
            const uint8_t h = (uint8_t)((3U * (uint16_t)prev_h + (uint16_t)v + 2U) >> 2);
            prev_h = h;
            if (smooth_w > 0) {
                const uint8_t sm = (uint8_t)((3U * (uint16_t)prev_row[x] + (uint16_t)h + 2U) >> 2);
                prev_row[x] = sm;
                out_gray[i] = sm;
            } else {
                out_gray[i] = h;
            }
        }
    }
#else
    for (int y = 0; y < out_h; ++y) {
#if FLOW_VIZ_REMOVE_ROW_BIAS
        const float row_base_mag = use_row_bias ? row_mag_base[y] : 0.0f;
#endif
        const int row_base = y * out_w;
        for (int x = 0; x < out_w; ++x) {
            const int i = row_base + x;
            float dx = 0.0f;
            float dy = 0.0f;
            read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);
#if FLOW_VIZ_REMOVE_GLOBAL_MOTION
            dx -= mean_dx;
            dy -= mean_dy;
#endif
            float mag = sqrtf(dx * dx + dy * dy);

            // D15: 使用固定增益 0.05 (mag=20px 时饱和)
            float mag_norm = mag * 0.05f;
            if (mag_norm > 1.0f) mag_norm = 1.0f;
            
            out_gray[i] = (uint8_t)(mag_norm * 255.0f + 0.5f);
        }
    }
#endif  // FLOW_VIZ_LIGHT_SMOOTH
#endif  // FLOW_VIZ_FIXED_SCALE
}

size_t flow_render_gray_to_jpeg(const uint8_t *gray,
                               int width,
                               int height,
                               uint8_t *jpeg_buf,
                               size_t jpeg_buf_size)
{
    if (gray == nullptr || jpeg_buf == nullptr || width <= 0 || height <= 0 || jpeg_buf_size < 256) {
        return 0;
    }

    static JPEG jpg;
    JPEGENCODE jpe;
    int rc = jpg.open(jpeg_buf, (int)jpeg_buf_size);
    if (rc != JPEG_SUCCESS) {
        return 0;
    }

    /* plan-007 H5: 尝试 JPEG_Q_BEST 排除 JPEG 块效应 */
    rc = jpg.encodeBegin(&jpe, width, height, JPEG_PIXEL_GRAYSCALE, JPEG_SUBSAMPLE_444, JPEG_Q_BEST);
    if (rc != JPEG_SUCCESS) {
        return 0;
    }

    const int pitch = width;
    const int iMCUCount = ((width + jpe.cx - 1) / jpe.cx) * ((height + jpe.cy - 1) / jpe.cy);
    for (int i = 0; i < iMCUCount && rc == JPEG_SUCCESS; i++) {
        rc = jpg.addMCU(&jpe, const_cast<uint8_t *>(&gray[jpe.x + jpe.y * width]), pitch);
    }

    if (rc != JPEG_SUCCESS) {
        return 0;
    }
    return (size_t)jpg.close();
}

static void hsv_to_rgb(float h, float s, float v, float *r, float *g, float *b)
{
    int i = (int)(h * 6.0f);
    float f = h * 6.0f - (float)i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);

    switch (i % 6) {
        case 0: *r = v; *g = t; *b = p; break;
        case 1: *r = q; *g = v; *b = p; break;
        case 2: *r = p; *g = v; *b = t; break;
        case 3: *r = p; *g = q; *b = v; break;
        case 4: *r = t; *g = p; *b = v; break;
        case 5: *r = v; *g = p; *b = q; break;
        default: *r = 0; *g = 0; *b = 0; break;
    }
}

void flow_render_to_rgb(uint8_t *out_rgb,
                        const int8_t *flow_data,
                        int out_w,
                        int out_h,
                        int out_stride,
                        int out_zp,
                        float out_scale)
{
    if (out_rgb == nullptr || flow_data == nullptr || out_w <= 0 || out_h <= 0 || out_stride < 2) {
        return;
    }

    const int pixels = out_w * out_h;

    for (int i = 0; i < pixels; ++i) {
        float dx = 0.0f;
        float dy = 0.0f;
        read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, pixels, i, &dx, &dy);

        const float mag = sqrtf(dx * dx + dy * dy);
        float angle = atan2f(dy, dx);
        if (angle < 0.0f) {
            angle += 2.0f * 3.14159265f;
        }
        const float hue = angle / (2.0f * 3.14159265f);

        float sat = 1.0f;
        float val = mag * 0.05f;  // D15: 降低增益，mag=20px 时饱和
        if (val > 1.0f) val = 1.0f;

        float r, g, b;
        hsv_to_rgb(hue, sat, val, &r, &g, &b);

        out_rgb[i * 3 + 0] = (uint8_t)(r * 255.0f);
        out_rgb[i * 3 + 1] = (uint8_t)(g * 255.0f);
        out_rgb[i * 3 + 2] = (uint8_t)(b * 255.0f);
    }
}

/**
 * D14: 分块 RGB 渲染 + JPEG 编码
 * 使用小 buffer（8 行）分块渲染，避免大内存分配
 * 返回 JPEG 大小，失败返回 0
 */
size_t flow_render_rgb_to_jpeg_block(const int8_t *flow_data,
                                      int out_w,
                                      int out_h,
                                      int out_stride,
                                      int out_zp,
                                      float out_scale,
                                      uint8_t *rgb_block,
                                      size_t rgb_block_size,
                                      uint8_t *jpeg_buf,
                                      size_t jpeg_buf_size)
{
    if (flow_data == nullptr || rgb_block == nullptr || jpeg_buf == nullptr ||
        out_w <= 0 || out_h <= 0 || jpeg_buf_size < 256) {
        return 0;
    }

    const int block_rows = 8;  // 每块 8 行
    const int pitch = out_w * 3;
    const size_t required_block_size = (size_t)block_rows * (size_t)pitch;
    if (rgb_block_size < required_block_size) {
        return 0;  // block buffer 太小
    }

    static JPEG jpg;
    JPEGENCODE jpe;
    int rc = jpg.open(jpeg_buf, (int)jpeg_buf_size);
    if (rc != JPEG_SUCCESS) {
        return 0;
    }

    rc = jpg.encodeBegin(&jpe, out_w, out_h, JPEG_PIXEL_RGB888, JPEG_SUBSAMPLE_444, JPEG_Q_BEST);
    if (rc != JPEG_SUCCESS) {
        return 0;
    }

    // 逐块渲染和编码
    for (int block_y = 0; block_y < out_h && rc == JPEG_SUCCESS; block_y += block_rows) {
        const int y_end = (block_y + block_rows < out_h) ? (block_y + block_rows) : out_h;
        const int rows_in_block = y_end - block_y;

        // 渲染这一块
        for (int local_y = 0; local_y < rows_in_block; ++local_y) {
            const int y = block_y + local_y;
            for (int x = 0; x < out_w; ++x) {
                const int i = y * out_w + x;
                float dx = 0.0f;
                float dy = 0.0f;
                read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, out_w * out_h, i, &dx, &dy);

                const float mag = sqrtf(dx * dx + dy * dy);
                float angle = atan2f(dy, dx);
                if (angle < 0.0f) {
                    angle += 2.0f * 3.14159265f;
                }
                const float hue = angle / (2.0f * 3.14159265f);

                float sat = 1.0f;
                float val = mag * 0.05f;  // D15: 降低增益
                if (val > 1.0f) val = 1.0f;

                float r, g, b;
                hsv_to_rgb(hue, sat, val, &r, &g, &b);

                const int row_off = local_y * pitch + x * 3;
                rgb_block[row_off + 0] = (uint8_t)(r * 255.0f);
                rgb_block[row_off + 1] = (uint8_t)(g * 255.0f);
                rgb_block[row_off + 2] = (uint8_t)(b * 255.0f);
            }
        }

        // 编码这一块中的所有 MCU
        // JPEG encoder 会跟踪 jpe.y 和 jpe.x，我们只需要按顺序提供数据
        while (jpe.y < y_end && rc == JPEG_SUCCESS) {
            if (jpe.y >= block_y && jpe.y < y_end) {
                const int local_y = jpe.y - block_y;
                rc = jpg.addMCU(&jpe, &rgb_block[local_y * pitch + jpe.x * 3], pitch);
            } else {
                // 跳过不在当前块的行（不应该发生）
                break;
            }
        }
    }

    if (rc != JPEG_SUCCESS) {
        return 0;
    }
    return (size_t)jpg.close();
}

size_t flow_render_rgb_to_jpeg(const uint8_t *rgb,
                               int width,
                               int height,
                               uint8_t *jpeg_buf,
                               size_t jpeg_buf_size)
{
    if (rgb == nullptr || jpeg_buf == nullptr || width <= 0 || height <= 0 || jpeg_buf_size < 256) {
        return 0;
    }

    static JPEG jpg;
    JPEGENCODE jpe;
    int rc = jpg.open(jpeg_buf, (int)jpeg_buf_size);
    if (rc != JPEG_SUCCESS) {
        return 0;
    }

    rc = jpg.encodeBegin(&jpe, width, height, JPEG_PIXEL_RGB888, JPEG_SUBSAMPLE_444, JPEG_Q_BEST);
    if (rc != JPEG_SUCCESS) {
        return 0;
    }

    const int pitch = width * 3;
    const int iMCUCount = ((width + jpe.cx - 1) / jpe.cx) * ((height + jpe.cy - 1) / jpe.cy);
    for (int i = 0; i < iMCUCount && rc == JPEG_SUCCESS; i++) {
        rc = jpg.addMCU(&jpe, const_cast<uint8_t *>(&rgb[jpe.y * width * 3 + jpe.x * 3]), pitch);
    }

    if (rc != JPEG_SUCCESS) {
        return 0;
    }
    return (size_t)jpg.close();
}

#include "flow_render.h"
#include "JPEGENC.h"
#include <math.h>
#include <stdlib.h>



static inline void read_flow_dxdy(const int8_t *flow_data,
                                  int out_stride,
                                  int out_zp,
                                  float out_scale,
                                  int i,
                                  float *dx,
                                  float *dy)
{
    const int qx = (int)flow_data[i * out_stride + 0];
    const int qy = (int)flow_data[i * out_stride + 1];
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

    for (int y = 0; y < out_h; ++y) {
        const int row_base = y * out_w;
        for (int x = 0; x < out_w; ++x) {
            const int i = row_base + x;
            float dx = 0.0f;
            float dy = 0.0f;
            read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, i, &dx, &dy);
            const float mag = sqrtf(dx * dx + dy * dy);

            // D15: 使用固定增益 0.05 (mag=20px 时饱和)
            float mag_norm = mag * 0.05f;
            if (mag_norm > 1.0f) mag_norm = 1.0f;
            
            out_gray[i] = (uint8_t)(mag_norm * 255.0f + 0.5f);
        }
    }
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
                read_flow_dxdy(flow_data, out_stride, out_zp, out_scale, i, &dx, &dy);

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



#include "flow_render.h"
#include "JPEGENC.h"
#include <math.h>
#include <stdlib.h>

// 1=固定 scale（plan-007 H2），0=per-frame max 归一化
#ifndef FLOW_VIZ_FIXED_SCALE
#define FLOW_VIZ_FIXED_SCALE 1
#endif

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

#if FLOW_VIZ_FIXED_SCALE
    const float kFixedScale = 80.0f;
    for (int i = 0; i < pixels; ++i) {
        const float dx = ((float)flow_data[i * out_stride + 0] - (float)out_zp) * out_scale;
        const float dy = ((float)flow_data[i * out_stride + 1] - (float)out_zp) * out_scale;
        const float mag = sqrtf(dx * dx + dy * dy);
        float v = mag * kFixedScale;
        if (v > 255.0f) {
            v = 255.0f;
        }
        out_gray[i] = (uint8_t)(v);
    }
#else
    float max_mag = 1e-6f;
    for (int i = 0; i < pixels; ++i) {
        const float dx = ((float)flow_data[i * out_stride + 0] - (float)out_zp) * out_scale;
        const float dy = ((float)flow_data[i * out_stride + 1] - (float)out_zp) * out_scale;
        const float mag = sqrtf(dx * dx + dy * dy);
        if (mag > max_mag) {
            max_mag = mag;
        }
    }

    for (int i = 0; i < pixels; ++i) {
        const float dx = ((float)flow_data[i * out_stride + 0] - (float)out_zp) * out_scale;
        const float dy = ((float)flow_data[i * out_stride + 1] - (float)out_zp) * out_scale;
        const float mag = sqrtf(dx * dx + dy * dy);
        float v = (max_mag > 1e-6f) ? (mag / max_mag) : 0.0f;
        if (v > 1.0f) {
            v = 1.0f;
        }
        out_gray[i] = (uint8_t)(v * 255.0f);
    }
#endif
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

#ifndef OPTICAL_CAM_OFLOW_FLOW_RENDER_H_
#define OPTICAL_CAM_OFLOW_FLOW_RENDER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 将光流 tensor (int8 dx,dy) 渲染为灰度图（magnitude -> 亮度）
 * 输入: out_data[H*W*2], 量化参数 out_zp, out_scale
 * 输出: gray[H*W], 0=无运动, 255=最大运动
 */
void flow_render_to_gray(uint8_t *out_gray,
                        const int8_t *flow_data,
                        int out_w,
                        int out_h,
                        int out_stride,
                        int out_zp,
                        float out_scale);

/**
 * 将灰度图编码为 JPEG，写入 jpeg_buf
 * 返回实际 JPEG 字节数，失败返回 0
 */
size_t flow_render_gray_to_jpeg(const uint8_t *gray,
                               int width,
                               int height,
                               uint8_t *jpeg_buf,
                               size_t jpeg_buf_size);

#ifdef __cplusplus
}
#endif

#endif  // OPTICAL_CAM_OFLOW_FLOW_RENDER_H_

#include "cam_input.h"

#include <string.h>
#include "img_proc_helium.h"

extern "C" {
#include "cisdp_sensor.h"
#include "hx_drv_timer.h"
#include "hx_drv_xdma.h"
#include "sensor_dp_lib.h"
}

#include "xprintf.h"

namespace {

constexpr uint32_t kFrameWaitRetry = 1000;
constexpr uint32_t kFrameWaitDelayMs = 2;
constexpr uint32_t kInterFrameDelayMs = 15;

// R10 result: keep BGR as default; RGB A/B did not improve flow readability.
#ifndef CAM_INPUT_USE_BGR
#define CAM_INPUT_USE_BGR 1
#endif

#ifndef CAM_INPUT_USE_HELIUM_RESIZE
#define CAM_INPUT_USE_HELIUM_RESIZE 0
#endif

static bool g_inited = false;
static bool g_first_frame_ready = false;
static uint32_t g_model_in_w = 0U;
static uint32_t g_model_in_h = 0U;

static int wait_new_frame()
{
    if (!g_first_frame_ready) {
        // 某些配置下首帧需要手动 retrigger 才会推进 first-frame flag。
        sensordplib_retrigger_capture();
        uint8_t firstframe_cap = 0;
        for (uint32_t i = 0; i < kFrameWaitRetry; ++i) {
            if (hx_drv_xdma_get_WDMA2FirstFrameCapflag(&firstframe_cap) == XDMA_NO_ERROR &&
                firstframe_cap == 1U) {
                g_first_frame_ready = true;
                break;
            }
            hx_drv_timer_cm55x_delay_ms(kFrameWaitDelayMs, TIMER_STATE_DC);
        }
        if (!g_first_frame_ready) {
            xprintf("wait first camera frame timeout, fallback to delay mode\n");
            hx_drv_timer_cm55x_delay_ms(kInterFrameDelayMs, TIMER_STATE_DC);
            g_first_frame_ready = true;
            return 0;
        }
        // 首帧就绪后再触发下一帧，保证首次输出是完整 frame。
        sensordplib_retrigger_capture();
        hx_drv_timer_cm55x_delay_ms(kInterFrameDelayMs, TIMER_STATE_DC);
        return 0;
    }

    sensordplib_retrigger_capture();
    hx_drv_timer_cm55x_delay_ms(kInterFrameDelayMs, TIMER_STATE_DC);
    return 0;
}

static int planar_to_rgb_model_input(uint8_t *dst, size_t dst_bytes)
{
    const uint32_t raw_w = app_get_raw_width();
    const uint32_t raw_h = app_get_raw_height();
    const uint32_t raw_c = app_get_raw_channels();
    const uint32_t raw_sz = app_get_raw_sz();
    const uint8_t *raw = reinterpret_cast<const uint8_t *>(app_get_raw_addr());

    if (dst == nullptr || raw == nullptr) {
        return -1;
    }
    if (raw_c != 3U || raw_w == 0U || raw_h == 0U) {
        xprintf("cam raw format unsupported: w=%u h=%u c=%u\n", raw_w, raw_h, raw_c);
        return -1;
    }
    if (g_model_in_w == 0U || g_model_in_h == 0U) {
        xprintf("cam model input size invalid: w=%u h=%u\n", g_model_in_w, g_model_in_h);
        return -1;
    }
    if (dst_bytes < static_cast<size_t>(g_model_in_w) * g_model_in_h * 3U) {
        return -1;
    }

    const uint32_t plane_sz = raw_w * raw_h;
    if (raw_sz < plane_sz * 3U) {
        xprintf("cam raw size invalid: sz=%u expect>=%u\n", raw_sz, plane_sz * 3U);
        return -1;
    }

#if CAM_INPUT_USE_HELIUM_RESIZE
    // Keep preprocessing aligned with official apps: planar BGR8U3C -> packed RGB24.
    float w_scale = 0.0f;
    float h_scale = 0.0f;
    if (g_model_in_w > 1U && g_model_in_h > 1U) {
        w_scale = (float)(raw_w - 1U) / (float)(g_model_in_w - 1U);
        h_scale = (float)(raw_h - 1U) / (float)(g_model_in_h - 1U);
    }
    hx_lib_image_resize_BGR8U3C_to_RGB24_helium((uint8_t *)raw,
                                                 dst,
                                                 (int)raw_w,
                                                 (int)raw_h,
                                                 (int)raw_c,
                                                 (int)g_model_in_w,
                                                 (int)g_model_in_h,
                                                 w_scale,
                                                 h_scale);
#if CAM_INPUT_USE_BGR
    // Export scripts use cv2.imread() (BGR), keep board input channel order consistent.
    const size_t pix_cnt = (size_t)g_model_in_w * (size_t)g_model_in_h;
    for (size_t i = 0; i < pix_cnt; ++i) {
        const size_t p = i * 3U;
        const uint8_t r = dst[p + 0U];
        dst[p + 0U] = dst[p + 2U];
        dst[p + 2U] = r;
    }
#endif
    return 0;
#else
    const uint8_t *plane_b = raw;
    const uint8_t *plane_g = raw + plane_sz;
    const uint8_t *plane_r = raw + plane_sz * 2U;

    // 针对 Optical Flow，我们需要保持最真实的像素物理属性，禁止随意拉伸（这会导致异常的光流向量）。
    // 最优法: 计算输入 (raw_w x raw_h) 在适应到输出 (g_model_in_w x g_model_in_h) 时，
    // 以中心对齐的方式，进行最大可能的等比裁剪 (Center Crop)
    
    // 计算缩放比例 (以更小的那一边为基准，确保填满目标区域)
    float scale = (float)raw_w / (float)g_model_in_w;
    float scale_h = (float)raw_h / (float)g_model_in_h;
    if (scale_h < scale) {
        scale = scale_h; 
    }

    // 计算中心裁剪框在 raw 下的尺寸和起始位置
    uint32_t crop_w = (uint32_t)(g_model_in_w * scale);
    uint32_t crop_h = (uint32_t)(g_model_in_h * scale);
    uint32_t start_x = (raw_w - crop_w) / 2U;
    uint32_t start_y = (raw_h - crop_h) / 2U;

    // Nearest Neighbor Resize with Center Crop
    for (uint32_t y = 0; y < g_model_in_h; ++y) {
        uint32_t src_y = start_y + (uint32_t)(y * scale);
        if (src_y >= raw_h) src_y = raw_h - 1U;
        
        for (uint32_t x = 0; x < g_model_in_w; ++x) {
            uint32_t src_x = start_x + (uint32_t)(x * scale);
            if (src_x >= raw_w) src_x = raw_w - 1U;
            
            const uint32_t src = src_y * raw_w + src_x;
            const uint32_t dst_idx = (y * g_model_in_w + x) * 3U;
            
            // Keep channel order configurable for quick A/B validation.
#if CAM_INPUT_USE_BGR
            dst[dst_idx + 0U] = plane_b[src];
            dst[dst_idx + 1U] = plane_g[src];
            dst[dst_idx + 2U] = plane_r[src];
#else
            dst[dst_idx + 0U] = plane_r[src];
            dst[dst_idx + 1U] = plane_g[src];
            dst[dst_idx + 2U] = plane_b[src];
#endif
        }
    }
    return 0;
#endif
}

static int capture_one(uint8_t *dst, size_t bytes_per_frame)
{
    if (wait_new_frame() != 0) {
        xprintf("wait new camera frame timeout\n");
        return -1;
    }
    return planar_to_rgb_model_input(dst, bytes_per_frame);
}

}  // namespace

int cam_input_init(uint32_t model_w, uint32_t model_h)
{
    if (g_inited) {
        return 0;
    }
    if (model_w == 0U || model_h == 0U) {
        xprintf("cam input model size invalid: w=%u h=%u\n", model_w, model_h);
        return -1;
    }
    g_model_in_w = model_w;
    g_model_in_h = model_h;

    if (cisdp_sensor_init() < 0) {
        xprintf("CIS Init fail\n");
        return -1;
    }
    if (cisdp_dp_init(true,
                      SENSORDPLIB_PATH_INT_INP_HW5X5_JPEG,
                      nullptr,
                      4,
                      APP_DP_RES_RGB640x480_INP_SUBSAMPLE_2X) < 0) {
        xprintf("DATAPATH Init fail\n");
        return -1;
    }

    cisdp_sensor_start();

    g_first_frame_ready = false;
    g_inited = true;
    xprintf("camera input init done\n");
    return 0;
}

int cam_input_get_frame(uint8_t *frame, size_t bytes_per_frame)
{
    if (!g_inited || frame == nullptr) {
        return -1;
    }
    return capture_one(frame, bytes_per_frame);
}

void cam_input_deinit(void)
{
    if (!g_inited) {
        return;
    }
    cisdp_sensor_stop();
    g_inited = false;
    g_first_frame_ready = false;
    g_model_in_w = 0U;
    g_model_in_h = 0U;
}

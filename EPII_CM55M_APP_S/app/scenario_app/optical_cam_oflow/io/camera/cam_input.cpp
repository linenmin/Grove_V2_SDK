#include "cam_input.h"

#include <string.h>

extern "C" {
#include "cisdp_sensor.h"
#include "hx_drv_timer.h"
#include "hx_drv_xdma.h"
}

#include "xprintf.h"

namespace {

constexpr uint32_t kModelInW = 240;
constexpr uint32_t kModelInH = 180;
constexpr uint32_t kFrameWaitRetry = 1000;
constexpr uint32_t kFrameWaitDelayMs = 2;
constexpr uint32_t kInterFrameDelayMs = 33;

static bool g_inited = false;
static bool g_pair_ready = false;
static bool g_first_frame_ready = false;

static int wait_new_frame()
{
    if (!g_first_frame_ready) {
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
            xprintf("wait first camera frame timeout\n");
            return -1;
        }
    }
    hx_drv_timer_cm55x_delay_ms(kInterFrameDelayMs, TIMER_STATE_DC);
    return 0;
}

static int planar320x240_to_rgb240x180(uint8_t *dst, size_t dst_bytes)
{
    const uint32_t raw_w = app_get_raw_width();
    const uint32_t raw_h = app_get_raw_height();
    const uint32_t raw_c = app_get_raw_channels();
    const uint32_t raw_sz = app_get_raw_sz();
    const uint8_t *raw = reinterpret_cast<const uint8_t *>(app_get_raw_addr());

    if (dst == nullptr || raw == nullptr) {
        return -1;
    }
    if (raw_c != 3U || raw_w < kModelInW || raw_h < kModelInH) {
        xprintf("cam raw format unsupported: w=%u h=%u c=%u\n", raw_w, raw_h, raw_c);
        return -1;
    }
    if (dst_bytes < static_cast<size_t>(kModelInW) * kModelInH * 3U) {
        return -1;
    }

    const uint32_t plane_sz = raw_w * raw_h;
    if (raw_sz < plane_sz * 3U) {
        xprintf("cam raw size invalid: sz=%u expect>=%u\n", raw_sz, plane_sz * 3U);
        return -1;
    }

    const uint8_t *plane_b = raw;
    const uint8_t *plane_g = raw + plane_sz;
    const uint8_t *plane_r = raw + plane_sz * 2U;

    const uint32_t crop_x = (raw_w - kModelInW) / 2U;
    const uint32_t crop_y = (raw_h - kModelInH) / 2U;

    for (uint32_t y = 0; y < kModelInH; ++y) {
        for (uint32_t x = 0; x < kModelInW; ++x) {
            const uint32_t src = (crop_y + y) * raw_w + (crop_x + x);
            const uint32_t dst_idx = (y * kModelInW + x) * 3U;
            dst[dst_idx + 0U] = plane_r[src];
            dst[dst_idx + 1U] = plane_g[src];
            dst[dst_idx + 2U] = plane_b[src];
        }
    }
    return 0;
}

static int capture_one(uint8_t *dst, size_t bytes_per_frame)
{
    if (wait_new_frame() != 0) {
        xprintf("wait new camera frame timeout\n");
        return -1;
    }
    return planar320x240_to_rgb240x180(dst, bytes_per_frame);
}

}  // namespace

int cam_input_init(void)
{
    if (g_inited) {
        return 0;
    }

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
    g_pair_ready = false;
    g_inited = true;
    xprintf("camera input init done\n");
    return 0;
}

int cam_input_get_frame_pair(uint8_t *frame_t, uint8_t *frame_t1, size_t bytes_per_frame)
{
    if (!g_inited || frame_t == nullptr || frame_t1 == nullptr) {
        return -1;
    }

    if (!g_pair_ready) {
        if (capture_one(frame_t, bytes_per_frame) != 0) {
            return -1;
        }
        if (capture_one(frame_t1, bytes_per_frame) != 0) {
            return -1;
        }
        g_pair_ready = true;
        return 0;
    }

    memcpy(frame_t, frame_t1, bytes_per_frame);
    return capture_one(frame_t1, bytes_per_frame);
}

void cam_input_deinit(void)
{
    if (!g_inited) {
        return;
    }
    cisdp_sensor_stop();
    g_inited = false;
    g_first_frame_ready = false;
    g_pair_ready = false;
}

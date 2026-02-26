#include "viz_publish.h"
#include <stdio.h>
#include <string.h>
#include "WE2_core.h"
#include "WE2_device.h"
#include "hx_drv_scu.h"
#include "spi_master_protocol.h"
#include "viz_uart.h"
#include "flow_render.h"
#include "cam_input.h"
#include "xprintf.h"
#include "common_config.h"

extern "C" {
#include "cisdp_sensor.h"
}

namespace {

constexpr uint32_t kMinJpegBytes = 128U;
constexpr uint32_t kMaxJpegBytes = 256U * 1024U;
constexpr uint32_t kJpegScanLen = 0x4B000U;
constexpr uint16_t kFallbackJpegW = 64U;
constexpr uint16_t kFallbackJpegH = 48U;

constexpr int kFlowVizMaxPixels = FLOW_MODEL_IN_W * FLOW_MODEL_IN_H;
constexpr int kFlowVizRgbBlockRows = 8;
constexpr size_t kFlowVizRgbBlockSize = kFlowVizRgbBlockRows * FLOW_MODEL_IN_W * 3;
constexpr size_t kFlowVizGrayJpegBufSize = 24576U;
constexpr size_t kFlowVizRgbJpegBufSize = 49152U;

} // namespace

__attribute__((section(".bss.NoInit"))) uint8_t g_flow_viz_gray[kFlowVizMaxPixels] __attribute__((aligned(32)));
__attribute__((section(".bss.NoInit"))) uint8_t g_flow_viz_jpeg[kFlowVizGrayJpegBufSize] __attribute__((aligned(32)));
__attribute__((section(".bss.NoInit"))) uint8_t g_flow_viz_rgb_block[kFlowVizRgbBlockSize] __attribute__((aligned(32)));

namespace {

static const uint8_t kFallbackInvokeJpeg[] = {
    0xff, 0xd8, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06,
    0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0a, 0x0c, 0x14, 0x0d,
    0x0c, 0x0b, 0x0b, 0x0c, 0x19, 0x12, 0x13, 0x0f, 0x14, 0x1d, 0x1a, 0x1f,
    0x1e, 0x1d, 0x1a, 0x1c, 0x1c, 0x20, 0x24, 0x2e, 0x27, 0x20, 0x22, 0x2c,
    0x23, 0x1c, 0x1c, 0x28, 0x37, 0x29, 0x2c, 0x30, 0x31, 0x34, 0x34, 0x34,
    0x1f, 0x27, 0x39, 0x3d, 0x38, 0x32, 0x3c, 0x2e, 0x33, 0x34, 0x32, 0xff,
    0xd9};

uint32_t g_viz_fail_cnt = 0;
uint32_t g_viz_skip_cnt = 0;
uint32_t g_last_good_jpeg_addr = 0U;
uint32_t g_last_good_jpeg_sz = 0U;

bool find_jpeg_payload(uint32_t base_addr,
                       uint32_t scan_len,
                       uint32_t *jpeg_addr,
                       uint32_t *jpeg_sz)
{
    if (base_addr == 0U || scan_len < 4U || jpeg_addr == nullptr || jpeg_sz == nullptr) {
        return false;
    }

    const uint8_t *buf = reinterpret_cast<const uint8_t *>(base_addr);
    for (uint32_t i = 0U; i + 1U < scan_len; ++i) {
        if (!(buf[i] == 0xFFU && buf[i + 1U] == 0xD8U)) {
            continue;
        }
        for (uint32_t j = i + 2U; j + 1U < scan_len; ++j) {
            if (buf[j] == 0xFFU && buf[j + 1U] == 0xD9U) {
                const uint32_t cur_sz = (j + 2U) - i;
                bool has_sos = false;
                for (uint32_t k = i + 2U; k + 1U < j; ++k) {
                    if (buf[k] == 0xFFU && buf[k + 1U] == 0xDAU) {
                        has_sos = true;
                        break;
                    }
                }
                if (has_sos && cur_sz >= kMinJpegBytes && cur_sz <= kMaxJpegBytes) {
                    *jpeg_addr = base_addr + i;
                    *jpeg_sz = cur_sz;
                    return true;
                }
                i = j;
                break;
            }
        }
    }
    return false;
}

bool is_jpeg_head_tail_ok(uint32_t jpeg_addr, uint32_t jpeg_sz)
{
    if (jpeg_addr == 0U || jpeg_sz < kMinJpegBytes || jpeg_sz > kMaxJpegBytes) {
        return false;
    }

    const uint8_t *buf = reinterpret_cast<const uint8_t *>(jpeg_addr);
    return (buf[0] == 0xFFU && buf[1] == 0xD8U && buf[2] == 0xFFU &&
            buf[jpeg_sz - 2U] == 0xFFU && buf[jpeg_sz - 1U] == 0xD9U);
}

} // namespace

extern "C" void publish_viz_payload(struct_yolov8_ob_algoResult *algo,
                                   uint32_t total_us,
                                   int loop_cnt,
                                   const int8_t *flow_data,
                                   int flow_w,
                                   int flow_h,
                                   int flow_stride,
                                   int flow_zp,
                                   float flow_scale,
                                   const int8_t *input_prev_q,
                                   int input_w,
                                   int input_h)
{
    (void)input_prev_q;
    (void)input_w;
    (void)input_h;

    viz_uart_poll_host_cmd();
    const uint8_t transport_mode = viz_uart_get_transport_mode();
    const bool send_uart = (transport_mode == 0U || transport_mode == 2U);
    const bool send_spi = (transport_mode == 1U || transport_mode == 2U);
    const bool send_raw = (transport_mode == 3U);
    const bool need_uart_invoke = send_uart;
    const uint32_t algo_tick_cycles =
        (total_us <= (UINT32_MAX / 400U)) ? (total_us * 400U) : UINT32_MAX;

#if !defined(FORCE_VIZ_CAMERA_JPEG) || (FORCE_VIZ_CAMERA_JPEG == 0)
    if (flow_data != nullptr && flow_w > 0 && flow_h > 0 &&
        flow_w * flow_h <= kFlowVizMaxPixels) {
        flow_render_to_gray(g_flow_viz_gray,
                          flow_data,
                          flow_w,
                          flow_h,
                          flow_stride,
                          flow_zp,
                          flow_scale);
        hx_CleanDCache_by_Addr((volatile void *)g_flow_viz_gray, (size_t)flow_w * (size_t)flow_h);
        size_t jpeg_sz = 0U;
#if FLOW_VIZ_RGB_OUTPUT
        jpeg_sz = flow_render_rgb_to_jpeg_block(
                           flow_data,
                           flow_w,
                           flow_h,
                           flow_stride,
                           flow_zp,
                           flow_scale,
                           g_flow_viz_rgb_block,
                           kFlowVizRgbBlockSize,
                            g_flow_viz_jpeg,
                            kFlowVizGrayJpegBufSize);
        if (jpeg_sz == 0U) {
            jpeg_sz = flow_render_gray_to_jpeg(g_flow_viz_gray,
                                                flow_w,
                                                flow_h,
                                                g_flow_viz_jpeg,
                                                kFlowVizGrayJpegBufSize);
        }
#else
        jpeg_sz = flow_render_gray_to_jpeg(g_flow_viz_gray,
                                                flow_w,
                                                flow_h,
                                                g_flow_viz_jpeg,
                                                kFlowVizGrayJpegBufSize);
#endif
        if (jpeg_sz > 0U) {
            hx_CleanDCache_by_Addr((volatile void *)g_flow_viz_jpeg, jpeg_sz);
            if (send_raw) {
                // RAW binary mode: skip Base64/JSON, send binary header + JPEG
                viz_uart_send_raw_jpeg(g_flow_viz_jpeg,
                                       jpeg_sz,
                                       (uint16_t)flow_w,
                                       (uint16_t)flow_h);
                return;
            }
            if (send_uart) {
                viz_uart_send_device_id_once();
                if (need_uart_invoke) {
                    viz_uart_send_invoke_jpeg(g_flow_viz_jpeg,
                                             jpeg_sz,
                                             (uint16_t)flow_w,
                                             (uint16_t)flow_h,
                                             algo_tick_cycles);
                }
            }
            if (send_spi && algo != nullptr) {
                const int jpg_ret = hx_drv_spi_mst_protocol_write_sp(
                    (uint32_t)g_flow_viz_jpeg, (uint32_t)jpeg_sz, DATA_TYPE_JPG);
                if (jpg_ret == 0) {
                    algo->algo_tick = total_us;
                    hx_drv_spi_mst_protocol_write_sp(
                        (uint32_t)algo, sizeof(struct_yolov8_ob_algoResult), DATA_TYPE_META_YOLOV8_OB_DATA);
                }
            }
            return;
        }
    }
#endif

    const uint32_t jpeg_base = app_get_jpeg_addr();
    uint32_t jpeg_addr = 0U;
    uint32_t jpeg_sz = 0U;
    uint32_t cisdp_jpeg_addr = 0U;
    uint32_t cisdp_jpeg_sz = 0U;
    uint32_t autofill_jpeg_sz = 0U;
    bool jpeg_ready = false;
    cisdp_get_jpginfo(&cisdp_jpeg_sz, &cisdp_jpeg_addr);
    jpeg_addr = cisdp_jpeg_addr;
    jpeg_sz = cisdp_jpeg_sz;

    if (jpeg_addr != 0U && jpeg_sz != 0U && jpeg_sz != 0xFFFFFFFFU && jpeg_sz <= kMaxJpegBytes) {
        hx_InvalidateDCache_by_Addr((volatile void *)jpeg_addr, jpeg_sz);
        if (is_jpeg_head_tail_ok(jpeg_addr, jpeg_sz)) {
            jpeg_ready = true;
        } else {
            uint32_t fixed_addr = 0U;
            uint32_t fixed_sz = 0U;
            const uint32_t local_scan_len = (jpeg_sz < kJpegScanLen) ? jpeg_sz : kJpegScanLen;
            if (find_jpeg_payload(jpeg_addr, local_scan_len, &fixed_addr, &fixed_sz)) {
                jpeg_addr = fixed_addr;
                jpeg_sz = fixed_sz;
                jpeg_ready = true;
            }
        }
    }

    if (!jpeg_ready && jpeg_base != 0U) {
        jpeg_addr = jpeg_base;
        autofill_jpeg_sz = app_get_jpeg_sz();
        jpeg_sz = autofill_jpeg_sz;
        if (jpeg_sz != 0U && jpeg_sz != 0xFFFFFFFFU && jpeg_sz <= kMaxJpegBytes) {
            hx_InvalidateDCache_by_Addr((volatile void *)jpeg_addr, jpeg_sz);
            if (is_jpeg_head_tail_ok(jpeg_addr, jpeg_sz)) {
                jpeg_ready = true;
            } else {
                uint32_t fixed_addr = 0U;
                uint32_t fixed_sz = 0U;
                const uint32_t local_scan_len = (jpeg_sz < kJpegScanLen) ? jpeg_sz : kJpegScanLen;
                if (find_jpeg_payload(jpeg_addr, local_scan_len, &fixed_addr, &fixed_sz)) {
                    jpeg_addr = fixed_addr;
                    jpeg_sz = fixed_sz;
                    jpeg_ready = true;
                }
            }
        }
    }

    if (!jpeg_ready && jpeg_base != 0U) {
        hx_InvalidateDCache_by_Addr((volatile void *)jpeg_base, kJpegScanLen);
        if (find_jpeg_payload(jpeg_base, kJpegScanLen, &jpeg_addr, &jpeg_sz)) {
            jpeg_ready = true;
        }
    }

    if (!jpeg_ready) {
        if (g_last_good_jpeg_addr != 0U && g_last_good_jpeg_sz != 0U) {
            jpeg_addr = g_last_good_jpeg_addr;
            jpeg_sz = g_last_good_jpeg_sz;
            jpeg_ready = true;
        }
    }

    if (!jpeg_ready) {
        if (send_uart) {
            viz_uart_send_device_id_once();
            if (need_uart_invoke) {
                viz_uart_send_invoke_jpeg(kFallbackInvokeJpeg,
                                          sizeof(kFallbackInvokeJpeg),
                                          kFallbackJpegW,
                                          kFallbackJpegH,
                                          algo_tick_cycles);
            }
        }
        if ((g_viz_skip_cnt % 20U) == 0U) {
            uint8_t sig0 = 0U, sig1 = 0U, sig2 = 0U, sig3 = 0U;
            if (jpeg_base != 0U) {
                hx_InvalidateDCache_by_Addr((volatile void *)jpeg_base, 16U);
                const uint8_t *sig = reinterpret_cast<const uint8_t *>(jpeg_base);
                sig0 = sig[0]; sig1 = sig[1]; sig2 = sig[2]; sig3 = sig[3];
            }
            xprintf("viz skip invalid jpeg addr=0x%x size=%u base=0x%x cisdp=0x%x/%u auto=%u sig=%02x%02x%02x%02x\n",
                    jpeg_addr, jpeg_sz, jpeg_base, cisdp_jpeg_addr, cisdp_jpeg_sz, autofill_jpeg_sz, sig0, sig1, sig2, sig3);
        }
        g_viz_skip_cnt++;
        return;
    }

    hx_InvalidateDCache_by_Addr((volatile void *)jpeg_addr, jpeg_sz);
    g_last_good_jpeg_addr = jpeg_addr;
    g_last_good_jpeg_sz = jpeg_sz;

    if (send_spi) {
        const int jpg_ret = hx_drv_spi_mst_protocol_write_sp(jpeg_addr, jpeg_sz, DATA_TYPE_JPG);
        if (jpg_ret != 0) {
            if ((g_viz_fail_cnt % 20U) == 0U) {
                xprintf("viz jpg tx fail ret=%d size=%u addr=0x%x\n", jpg_ret, jpeg_sz, jpeg_addr);
            }
            g_viz_fail_cnt++;
        } else if (algo != nullptr) {
            algo->algo_tick = total_us;
            hx_drv_spi_mst_protocol_write_sp((uint32_t)algo, sizeof(struct_yolov8_ob_algoResult), DATA_TYPE_META_YOLOV8_OB_DATA);
        }
    }

    if (send_uart) {
        viz_uart_send_device_id_once();
        if (need_uart_invoke) {
            viz_uart_send_invoke_jpeg(reinterpret_cast<const uint8_t *>(jpeg_addr),
                                      jpeg_sz,
                                      app_get_raw_width(),
                                      app_get_raw_height(),
                                      algo_tick_cycles);
        }
    }
}

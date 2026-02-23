/*
 * cvapp.cpp
 *
 *  Created on: 2018
 *      Author: 902452
 */

#include <assert.h>
#include <forward_list>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <string.h>

#include "WE2_core.h"
#include "WE2_device.h"
#include "board.h"
#include "cvapp_yolov8n_ob.h"
#include "ethosu_driver.h"
#include "hx_drv_gpio.h"
#include "hx_drv_jpeg.h"
#include "hx_drv_scu.h"
#include "cam_input.h"
#include "memory_manage.h"
#include "ob_debug_stats.h"
#include "ob_perf.h"
#include "ob_runtime_ctx.h"
#include "viz_uart.h"
#include "flow_render.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "xprintf.h"

#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif

#define INPUT_IMAGE_CHANNELS 6
#define YOLOV8_OB_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS

#ifdef TRUSTZONE_SEC
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE BASE_ADDR_APB_U55_CTRL
#endif
#endif

using namespace std;

extern "C" {
void SSPI_CS_GPIO_Output_Level(bool setLevelHigh)
{
    hx_drv_gpio_set_out_value(GPIO16, (GPIO_OUT_LEVEL_E)setLevelHigh);
}

void SSPI_CS_GPIO_Pinmux(bool setGpioFn)
{
    if (setGpioFn) {
        hx_drv_scu_set_PB5_pinmux(SCU_PB5_PINMUX_GPIO16, 0);
    } else {
        hx_drv_scu_set_PB5_pinmux(SCU_PB5_PINMUX_SPI_M_CS_1, 0);
    }
}

void SSPI_CS_GPIO_Dir(bool setDirOut)
{
    if (setDirOut) {
        hx_drv_gpio_set_output(GPIO16, GPIO_OUT_HIGH);
    } else {
        hx_drv_gpio_set_input(GPIO16);
    }
}
}

extern "C" {
#include "cisdp_sensor.h"
#include "spi_master_protocol.h"
}

namespace {

// 兼顾 AllocateTensors 与 camera WDMA 动态缓冲。阶段 D 增加 flow viz 缓冲 64KB，arena 减 64KB 以保留 WDMA 空间。
constexpr int tensor_arena_size = 1616 * 1024;

static uint32_t tensor_arena = 0;
static ethosu_driver ethosu_drv;

static tflite::MicroInterpreter *yolov8n_ob_int_ptr = nullptr;
static TfLiteTensor *yolov8n_ob_input = nullptr;
static TfLiteTensor *yolov8n_ob_output = nullptr;
static int g_model_in_w = 0;
static int g_model_in_h = 0;
static int g_model_in_c = 0;
static int g_model_out_w = 0;
static int g_model_out_h = 0;
static int g_model_out_c = 0;
static size_t g_raw_frame_bytes = 0U;
static uint32_t g_viz_fail_cnt = 0;
static uint32_t g_viz_skip_cnt = 0;
static uint32_t g_last_good_jpeg_addr = 0U;
static uint32_t g_last_good_jpeg_sz = 0U;
static bool g_prev_frame_valid = false;
// 4X 子采样下的 160x120 JPEG 可能小于 1KB，阈值需放宽。
constexpr uint32_t kMinJpegBytes = 128U;
constexpr uint32_t kMaxJpegBytes = 256U * 1024U;
constexpr uint32_t kJpegScanLen = 0x4B000U;
constexpr uint16_t kFallbackJpegW = 64U;
constexpr uint16_t kFallbackJpegH = 48U;
// 光流渲染缓冲：176x224 灰度 + 24KB JPEG（最小化以保留 WDMA 内存）
constexpr int kFlowVizMaxPixels = 176 * 224;
constexpr size_t kFlowVizJpegBufSize = 24576U;
__attribute__((section(".bss.NoInit"))) static uint8_t g_flow_viz_gray[kFlowVizMaxPixels] __attribute__((aligned(32)));
__attribute__((section(".bss.NoInit"))) static uint8_t g_flow_viz_jpeg[kFlowVizJpegBufSize] __attribute__((aligned(32)));
static const uint8_t kFallbackInvokeJpeg[] = {
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0a, 0x0c, 0x14, 0x0d, 0x0c, 0x0b, 0x0b, 0x0c, 0x19, 0x12,
    0x13, 0x0f, 0x14, 0x1d, 0x1a, 0x1f, 0x1e, 0x1d, 0x1a, 0x1c, 0x1c, 0x20,
    0x24, 0x2e, 0x27, 0x20, 0x22, 0x2c, 0x23, 0x1c, 0x1c, 0x28, 0x37, 0x29,
    0x2c, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1f, 0x27, 0x39, 0x3d, 0x38, 0x32,
    0x3c, 0x2e, 0x33, 0x34, 0x32, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x09, 0x09,
    0x09, 0x0c, 0x0b, 0x0c, 0x18, 0x0d, 0x0d, 0x18, 0x32, 0x21, 0x1c, 0x21,
    0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32,
    0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32,
    0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32,
    0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32,
    0x32, 0x32, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x30, 0x00, 0x40, 0x03,
    0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xc4, 0x00,
    0x1b, 0x00, 0x01, 0x01, 0x00, 0x02, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x04, 0x06, 0x02, 0x03,
    0x05, 0x08, 0xff, 0xc4, 0x00, 0x28, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03,
    0x03, 0x04, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x02, 0x00, 0x03, 0x04, 0x11, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x07,
    0x51, 0x61, 0x81, 0x13, 0x15, 0x23, 0x71, 0xd1, 0xff, 0xc4, 0x00, 0x1b,
    0x01, 0x00, 0x02, 0x02, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x05, 0x07, 0x01, 0x02, 0x03,
    0x06, 0xff, 0xc4, 0x00, 0x1f, 0x11, 0x00, 0x02, 0x02, 0x02, 0x02, 0x03,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x02, 0x03, 0x04, 0x11, 0x31, 0x41, 0x05, 0x12, 0x13, 0x32, 0xff, 0xda,
    0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00,
    0xf2, 0x62, 0x22, 0x79, 0xe2, 0xc9, 0x11, 0x39, 0x53, 0xa6, 0xf5, 0xaa,
    0xa5, 0x2a, 0x48, 0xcf, 0x51, 0xd8, 0x2a, 0xa2, 0x8c, 0x96, 0x27, 0x80,
    0x07, 0x93, 0x2a, 0x3d, 0x3f, 0xe9, 0xbd, 0x95, 0x2b, 0x45, 0xad, 0xad,
    0x2b, 0x57, 0xb9, 0x75, 0xde, 0x82, 0xd4, 0x21, 0x29, 0xf1, 0x81, 0x95,
    0xdc, 0xb0, 0xf7, 0xce, 0x37, 0xf3, 0x8c, 0xce, 0xb5, 0x53, 0x2b, 0x5e,
    0xa2, 0x29, 0x97, 0x9b, 0x56, 0x2c, 0x7d, 0xac, 0x7c, 0xf5, 0xd9, 0x2c,
    0x89, 0x5c, 0xd5, 0xfd, 0x37, 0xd2, 0x6e, 0xad, 0x1b, 0xf5, 0xaa, 0xd6,
    0x77, 0x2a, 0xa7, 0xb0, 0x9a, 0x8c, 0xc8, 0xc7, 0x6d, 0x9b, 0x39, 0x38,
    0xdb, 0x91, 0xef, 0xe7, 0x89, 0x2a, 0xbc, 0xb3, 0xb8, 0xd3, 0xee, 0xea,
    0xda, 0x5d, 0xd2, 0x6a, 0x55, 0xe9, 0x37, 0x6b, 0xa3, 0x72, 0x3f, 0xd1,
    0xf3, 0xe6, 0x66, 0xda, 0x27, 0x57, 0xe8, 0xd7, 0x13, 0x3e, 0x9c, 0xa4,
    0xfe, 0x7c, 0xae, 0x9f, 0x27, 0x44, 0x44, 0x4e, 0x23, 0xa6, 0x4c, 0x44,
    0x47, 0xca, 0x1c, 0xd8, 0x3a, 0x26, 0x9a, 0x55, 0xeb, 0x0d, 0x3d, 0x6a,
    0x22, 0xb8, 0x05, 0xd8, 0x06, 0x19, 0xdc, 0x23, 0x10, 0x7e, 0x88, 0x07,
    0xea, 0x59, 0xe7, 0xcf, 0x96, 0xf5, 0xea, 0x5a, 0xdc, 0xd2, 0xb8, 0xa2,
    0xdd, 0xb5, 0x69, 0x38, 0x74, 0x6c, 0x03, 0x86, 0x07, 0x20, 0xef, 0x2b,
    0x7a, 0x0f, 0x5b, 0xe9, 0xba, 0x9d, 0xa2, 0x8b, 0xca, 0xf4, 0x6c, 0xef,
    0x14, 0x7f, 0x22, 0x54, 0x6e, 0xd4, 0x38, 0xc6, 0xea, 0xc7, 0x6d, 0xf3,
    0xc6, 0x73, 0xcf, 0x38, 0xcc, 0x66, 0x89, 0xa4, 0xb4, 0xc9, 0x7f, 0x1b,
    0x7c, 0x23, 0x17, 0x5c, 0x9e, 0x9f, 0x26, 0xd1, 0x25, 0x3e, 0xa4, 0xd3,
    0x44, 0xea, 0x5a, 0x4c, 0xa8, 0xaa, 0x5e, 0xd5, 0x59, 0xc8, 0x18, 0xee,
    0x3d, 0xcc, 0x32, 0x7d, 0xf6, 0x00, 0x7d, 0x09, 0xbb, 0x6a, 0xbd, 0x65,
    0xa3, 0x69, 0x96, 0x8d, 0x51, 0x2e, 0xe8, 0xdd, 0xd6, 0x20, 0xfe, 0x3a,
    0x34, 0x1c, 0x3f, 0x71, 0xdb, 0x62, 0x46, 0x42, 0xf3, 0xe7, 0xe7, 0x19,
    0xe2, 0x48, 0xf5, 0x2d, 0x46, 0xe3, 0x56, 0xd4, 0x6b, 0x5f, 0x5d, 0x15,
    0x35, 0xaa, 0x9c, 0x9e, 0xd1, 0x80, 0x30, 0x30, 0x00, 0xfe, 0x80, 0x02,
    0x66, 0xf9, 0xad, 0x68, 0xdf, 0xc8, 0xdf, 0x07, 0x0f, 0x9a, 0x7b, 0x66,
    0x24, 0x44, 0x45, 0x48, 0x53, 0xc5, 0x88, 0x89, 0x67, 0x12, 0x02, 0x22,
    0x20, 0x02, 0x22, 0x20, 0x02, 0x22, 0x20, 0x07, 0xff, 0xd9};

static ob_runtime_ctx_t g_ctx = {};

static void quantize_rgb_frame_inplace(const uint8_t *src_raw,
                                       int8_t *dst_q,
                                       size_t bytes_per_frame)
{
    if (src_raw == nullptr || dst_q == nullptr) {
        return;
    }
    for (size_t i = 0; i < bytes_per_frame; ++i) {
        int16_t q = (int16_t)src_raw[i] - 128;
        if (q > 127) {
            q = 127;
        }
        if (q < -128) {
            q = -128;
        }
        dst_q[i] = (int8_t)q;
    }
}

static void compute_checksum_from_q(const int8_t *buf_q, size_t len, ob_checksum_stats_t *stats)
{
    if (buf_q == nullptr || stats == nullptr || len == 0U) {
        return;
    }

    uint32_t sum = 0U;
    uint8_t min_v = 0xFFU;
    uint8_t max_v = 0U;
    for (size_t i = 0; i < len; ++i) {
        const int16_t restored = (int16_t)buf_q[i] + 128;
        const uint8_t v = (restored < 0) ? 0U : (restored > 255 ? 255U : (uint8_t)restored);
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

static void _arm_npu_irq_handler(void)
{
    ethosu_irq_handler(&ethosu_drv);
}

static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);
    NVIC_EnableIRQ(ethosu_irqnum);
}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;

    _arm_npu_irq_init();

#if TFLM2209_U55TAG2205
    const void *ethosu_base_address = (void *)(U55_BASE);
#else
    void *const ethosu_base_address = (void *)(U55_BASE);
#endif

    err = ethosu_init(&ethosu_drv,
                      ethosu_base_address,
                      NULL,
                      0,
                      security_enable,
                      privilege_enable);
    if (err != 0) {
        xprintf("failed to initalise Ethos-U device\n");
        return err;
    }

    xprintf("Ethos-U55 device initialised\n");
    return 0;
}

static bool find_jpeg_payload(uint32_t base_addr,
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

static bool is_jpeg_head_tail_ok(uint32_t jpeg_addr, uint32_t jpeg_sz)
{
    if (jpeg_addr == 0U || jpeg_sz < kMinJpegBytes || jpeg_sz > kMaxJpegBytes) {
        return false;
    }

    const uint8_t *buf = reinterpret_cast<const uint8_t *>(jpeg_addr);
    return (buf[0] == 0xFFU && buf[1] == 0xD8U && buf[2] == 0xFFU &&
            buf[jpeg_sz - 2U] == 0xFFU && buf[jpeg_sz - 1U] == 0xD9U);
}

static void publish_viz_payload(struct_yolov8_ob_algoResult *algo,
                               uint32_t total_us,
                               int loop_cnt,
                               const int8_t *flow_data,
                               int flow_w,
                               int flow_h,
                               int flow_stride,
                               int flow_zp,
                               float flow_scale)
{
    viz_uart_poll_host_cmd();
    const uint8_t transport_mode = viz_uart_get_transport_mode();
    const bool send_uart = (transport_mode == 0U || transport_mode == 2U);
    const bool send_spi = (transport_mode == 1U || transport_mode == 2U);
    const bool need_uart_invoke = send_uart;
    const uint32_t algo_tick_cycles =
        (total_us <= (UINT32_MAX / 400U)) ? (total_us * 400U) : UINT32_MAX;

    // 阶段 D：有 flow 输出时优先发送光流渲染图
    // FORCE_VIZ_CAMERA_JPEG：强制走摄像头分支，用于 agent 可见调试闭环（plan-008）
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
        const size_t jpeg_sz = flow_render_gray_to_jpeg(g_flow_viz_gray,
                                                       flow_w,
                                                       flow_h,
                                                       g_flow_viz_jpeg,
                                                       kFlowVizJpegBufSize);
        if (jpeg_sz > 0U) {
            hx_CleanDCache_by_Addr((volatile void *)g_flow_viz_jpeg, jpeg_sz);
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
        // 优先使用驱动返回的 frame-aware jpeg addr/size，避免 autofill 尺寸抖动。
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
        // 退回 autofill size，尝试局部修复。
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
        // autofill 失效时，兜底扫描 WDMA2 缓冲区中的 JPEG 片段。
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
            // 即使当前帧 JPEG 无效，也先维持握手输出，便于网页侧完成模式同步。
            viz_uart_send_device_id_once();
            if (need_uart_invoke) {
                // 先保证 Himax HTML 可视化链路可见，再并行定位真实 JPEG 路径问题。
                viz_uart_send_invoke_jpeg(kFallbackInvokeJpeg,
                                          sizeof(kFallbackInvokeJpeg),
                                          kFallbackJpegW,
                                          kFallbackJpegH,
                                          algo_tick_cycles);
            }
        }
        if ((g_viz_skip_cnt % 20U) == 0U) {
            uint8_t sig0 = 0U;
            uint8_t sig1 = 0U;
            uint8_t sig2 = 0U;
            uint8_t sig3 = 0U;
            if (jpeg_base != 0U) {
                hx_InvalidateDCache_by_Addr((volatile void *)jpeg_base, 16U);
                const uint8_t *sig = reinterpret_cast<const uint8_t *>(jpeg_base);
                sig0 = sig[0];
                sig1 = sig[1];
                sig2 = sig[2];
                sig3 = sig[3];
            }
            xprintf("viz skip invalid jpeg addr=0x%x size=%u base=0x%x cisdp=0x%x/%u auto=%u sig=%02x%02x%02x%02x\n",
                    jpeg_addr,
                    jpeg_sz,
                    jpeg_base,
                    cisdp_jpeg_addr,
                    cisdp_jpeg_sz,
                    autofill_jpeg_sz,
                    sig0,
                    sig1,
                    sig2,
                    sig3);
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
            const int meta_ret = hx_drv_spi_mst_protocol_write_sp(
                (uint32_t)algo, sizeof(struct_yolov8_ob_algoResult), DATA_TYPE_META_YOLOV8_OB_DATA);
            if (meta_ret != 0) {
                if ((g_viz_fail_cnt % 20U) == 0U) {
                    xprintf("viz meta tx fail ret=%d\n", meta_ret);
                }
                g_viz_fail_cnt++;
            }
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

}  // namespace

int cv_yolov8n_ob_init(bool security_enable, bool privilege_enable, uint32_t model_addr)
{
    int ercode = 0;

    g_ctx.loop_cnt = 0;
#ifdef VIZ_UART_MODE
    // VIZ 模式下保留低频性能与亮度统计，支撑阶段 A 调优。
    g_ctx.log_print_interval = 20;
#else
    g_ctx.log_print_interval = 5;
#endif

    tensor_arena = mm_reserve_align(tensor_arena_size, 0x20);
    if (tensor_arena == 0) {
        xprintf("alloc tensor arena fail, size=%d\n", tensor_arena_size);
        return -1;
    }
    xprintf("TA[%x], size=%d\r\n", tensor_arena, tensor_arena_size);

    if (_arm_npu_init(security_enable, privilege_enable) != 0) {
        return -1;
    }

    // 计时初始化放在启动阶段，保证 run 中只负责打点。
    ob_perf_init();

    if (model_addr != 0) {
        static const tflite::Model *yolov8n_ob_model =
            tflite::GetModel((const void *)model_addr);

        if (yolov8n_ob_model->version() != TFLITE_SCHEMA_VERSION) {
            xprintf("[ERROR] model schema %d != %d\n",
                    yolov8n_ob_model->version(),
                    TFLITE_SCHEMA_VERSION);
            return -1;
        }
        xprintf("model schema %d\n", yolov8n_ob_model->version());

#if TFLM2209_U55TAG2205
        static tflite::MicroErrorReporter yolov8n_ob_micro_error_reporter;
#endif
        static tflite::MicroMutableOpResolver<2> yolov8n_ob_op_resolver;

        yolov8n_ob_op_resolver.AddTranspose();
        if (kTfLiteOk != yolov8n_ob_op_resolver.AddEthosU()) {
            xprintf("Failed to add Arm NPU support to op resolver.");
            return false;
        }

#if TFLM2209_U55TAG2205
        static tflite::MicroInterpreter yolov8n_ob_static_interpreter(
            yolov8n_ob_model,
            yolov8n_ob_op_resolver,
            (uint8_t *)tensor_arena,
            tensor_arena_size,
            &yolov8n_ob_micro_error_reporter);
#else
        static tflite::MicroInterpreter yolov8n_ob_static_interpreter(
            yolov8n_ob_model,
            yolov8n_ob_op_resolver,
            (uint8_t *)tensor_arena,
            tensor_arena_size);
#endif

        if (yolov8n_ob_static_interpreter.AllocateTensors() != kTfLiteOk) {
            xprintf("AllocateTensors fail, arena=%d\n", tensor_arena_size);
            return false;
        }

        yolov8n_ob_int_ptr = &yolov8n_ob_static_interpreter;
        yolov8n_ob_input = yolov8n_ob_static_interpreter.input(0);
        yolov8n_ob_output = yolov8n_ob_static_interpreter.output(0);

        if (yolov8n_ob_input == nullptr || yolov8n_ob_output == nullptr ||
            yolov8n_ob_input->dims == nullptr || yolov8n_ob_output->dims == nullptr) {
            xprintf("input/output tensor metadata missing\n");
            return -1;
        }

        const TfLiteIntArray *in_dims = yolov8n_ob_input->dims;
        if (in_dims->size != 4) {
            xprintf("input dims size unsupported: %d\n", in_dims->size);
            return -1;
        }
        g_model_in_h = in_dims->data[1];
        g_model_in_w = in_dims->data[2];
        g_model_in_c = in_dims->data[3];
        if (g_model_in_w <= 0 || g_model_in_h <= 0 || g_model_in_c != INPUT_IMAGE_CHANNELS) {
            xprintf("input dims invalid: h=%d w=%d c=%d expect_c=%d\n",
                    g_model_in_h,
                    g_model_in_w,
                    g_model_in_c,
                    INPUT_IMAGE_CHANNELS);
            return -1;
        }

        const TfLiteIntArray *out_dims = yolov8n_ob_output->dims;
        if (out_dims->size < 3) {
            xprintf("output dims size unsupported: %d\n", out_dims->size);
            return -1;
        }
        if (out_dims->size == 4) {
            g_model_out_h = out_dims->data[1];
            g_model_out_w = out_dims->data[2];
            g_model_out_c = out_dims->data[3];
        } else {
            g_model_out_h = out_dims->data[0];
            g_model_out_w = out_dims->data[1];
            g_model_out_c = out_dims->data[2];
        }
        if (g_model_out_w <= 0 || g_model_out_h <= 0 || g_model_out_c < 2) {
            xprintf("output dims invalid: h=%d w=%d c=%d\n",
                    g_model_out_h,
                    g_model_out_w,
                    g_model_out_c);
            return -1;
        }

        g_raw_frame_bytes = (size_t)g_model_in_w * (size_t)g_model_in_h * 3U;
        xprintf("model io: in(h=%d,w=%d,c=%d) out(h=%d,w=%d,c=%d)\n",
                g_model_in_h,
                g_model_in_w,
                g_model_in_c,
                g_model_out_h,
                g_model_out_w,
                g_model_out_c);
    }

    g_prev_frame_valid = false;

    if (cam_input_init((uint32_t)g_model_in_w, (uint32_t)g_model_in_h) != 0) {
        xprintf("camera init fail\n");
        return -1;
    }

    xprintf("initial done\n");
    return ercode;
}

int cv_yolov8n_ob_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob)
{
    int ercode = 0;
    memset(algoresult_yolov8n_ob, 0, sizeof(struct_yolov8_ob_algoResult));

    if (yolov8n_ob_int_ptr == nullptr) {
        return ercode;
    }
    ob_perf_stamp_t t_total_start;
    ob_perf_stamp_t t_total_end;
    ob_perf_stamp_t t_io_start;
    ob_perf_stamp_t t_io_end;
    ob_perf_stamp_t t_preproc_start;
    ob_perf_stamp_t t_preproc_end;
    ob_perf_stamp_t t_infer_start;
    ob_perf_stamp_t t_infer_end;

    ob_perf_mark(&t_total_start);
    ob_perf_mark(&t_io_start);

    const size_t pix_cnt = (size_t)g_model_in_w * (size_t)g_model_in_h;
    int8_t *input_ptr = (int8_t *)yolov8n_ob_input->data.data;
    int8_t *curr_q = input_ptr + (pix_cnt * 3U);
    uint8_t *curr_raw = reinterpret_cast<uint8_t *>(curr_q);

    if (cam_input_get_frame(curr_raw, g_raw_frame_bytes) != 0) {
        xprintf("camera frame capture fail\n");
        return -1;
    }
    ob_perf_mark(&t_io_end);
    ob_perf_mark(&t_preproc_start);

    ob_compute_checksum(curr_raw, g_raw_frame_bytes, &g_ctx.raw2_stats);
    quantize_rgb_frame_inplace(curr_raw, curr_q, g_raw_frame_bytes);

    if (!g_prev_frame_valid) {
        memcpy(input_ptr, curr_q, g_raw_frame_bytes);
        g_prev_frame_valid = true;
        return 0;
    }

    // 上一帧始终缓存于 input 前 3 通道，避免额外占用 mm_reserve 内存。
    compute_checksum_from_q(input_ptr, g_raw_frame_bytes, &g_ctx.raw1_stats);

    ob_perf_mark(&t_preproc_end);
    ob_perf_mark(&t_infer_start);

    const TfLiteStatus invoke_status = yolov8n_ob_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        xprintf("optical flow invoke fail\n");
        return -1;
    }

    memcpy(input_ptr, curr_q, g_raw_frame_bytes);

    ob_perf_mark(&t_infer_end);
    ob_perf_mark(&t_total_end);

    g_ctx.sd_us = ob_perf_elapsed_us(&t_io_start, &t_io_end);
    g_ctx.preproc_us = ob_perf_elapsed_us(&t_preproc_start, &t_preproc_end);
    g_ctx.infer_us = ob_perf_elapsed_us(&t_infer_start, &t_infer_end);
    g_ctx.total_us = ob_perf_elapsed_us(&t_total_start, &t_total_end);

    const float out_scale =
        ((TfLiteAffineQuantization *)(yolov8n_ob_output->quantization.params))->scale->data[0];
    const int out_zp =
        ((TfLiteAffineQuantization *)(yolov8n_ob_output->quantization.params))->zero_point->data[0];
    const int8_t *out_data = yolov8n_ob_output->data.int8;

    publish_viz_payload(algoresult_yolov8n_ob,
                       g_ctx.total_us,
                       g_ctx.loop_cnt,
                       out_data,
                       g_model_out_w,
                       g_model_out_h,
                       g_model_out_c,
                       out_zp,
                       out_scale);

    ob_flow_summary_t flow_summary = {};
    ob_compute_flow_summary(out_data,
                            g_model_out_w,
                            g_model_out_h,
                            g_model_out_c,
                            out_zp,
                            out_scale,
                            &flow_summary);

        if (ob_should_log(g_ctx.loop_cnt, g_ctx.log_print_interval)) {
        ob_log_infer_line(g_ctx.loop_cnt,
                          0,
                          0,
                          &flow_summary,
                          &g_ctx.raw1_stats,
                          &g_ctx.raw2_stats,
                          g_ctx.sd_us,
                          g_ctx.preproc_us,
                          g_ctx.infer_us,
                          g_ctx.total_us);
    }

    g_ctx.loop_cnt++;
    return ercode;
}

int cv_yolov8n_ob_deinit()
{
    cam_input_deinit();
    return 0;
}

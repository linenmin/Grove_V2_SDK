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
#include "ff.h"

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

// 兼顾 AllocateTensors 与 camera/mm_reserve 动态缓冲。
// D8 结论：多尺度 vela 模型需要至少 1432KB，Vela 报告 sram_memory_used=1430KB
// D14: 保持 1432KB，通过优化 JPEG buffer 大小释放内存
#ifndef FLOW_TENSOR_ARENA_KB
#define FLOW_TENSOR_ARENA_KB 1432
#endif
constexpr int tensor_arena_size = FLOW_TENSOR_ARENA_KB * 1024;

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
static int8_t *g_curr_q_shadow = nullptr;
// prev_q 缓冲区：存储上一帧量化数据，用于 NHWC 6 通道交错拼装
static int8_t *g_prev_q_buffer = nullptr;
static uint32_t g_viz_fail_cnt = 0;
static uint32_t g_viz_skip_cnt = 0;
static uint32_t g_last_good_jpeg_addr = 0U;
static uint32_t g_last_good_jpeg_sz = 0U;
static bool g_prev_frame_valid = false;
// R15-P2: 开启后固定使用一组(prev,curr)输入帧对，便于离线逐元素对齐。
// 生产环境需关闭，让每帧都使用新的摄像头输入
#ifndef FLOW_DBG_FREEZE_PAIR
#define FLOW_DBG_FREEZE_PAIR 0
#endif
// R15-P3: 从 SD 读取固定量化帧对（prev/curr），用于和离线脚本同输入逐元素对齐。
// 失败时自动回退相机链路，不阻断主流程。
#ifndef FLOW_DBG_OFFLINE_INJECT
#define FLOW_DBG_OFFLINE_INJECT 0
#endif
#ifndef FLOW_DBG_OFFLINE_PREV_Q_PATH
#define FLOW_DBG_OFFLINE_PREV_Q_PATH "0:/ai_master/oflow_debug/prev_q_int8_150x200x3.raw"
#endif
#ifndef FLOW_DBG_OFFLINE_CURR_Q_PATH
#define FLOW_DBG_OFFLINE_CURR_Q_PATH "0:/ai_master/oflow_debug/curr_q_int8_150x200x3.raw"
#endif
// R15-P4(no-SD): 无 SD 卡时使用确定性合成帧对，保证板端同输入可复现。
// 现在使用真实摄像头输入，将合成注入关掉
#ifndef FLOW_DBG_SYNTH_INJECT
#define FLOW_DBG_SYNTH_INJECT 0
#endif
#ifndef FLOW_DBG_SYNTH_SHIFT_X
#define FLOW_DBG_SYNTH_SHIFT_X 3
#endif
#ifndef FLOW_DBG_SYNTH_SHIFT_Y
#define FLOW_DBG_SYNTH_SHIFT_Y 1
#endif
#ifndef FLOW_DBG_SYNTH_CURR_CONST
#define FLOW_DBG_SYNTH_CURR_CONST 0
#endif
#ifndef FLOW_DBG_SYNTH_PREV_CONST
#define FLOW_DBG_SYNTH_PREV_CONST 0
#endif
// R28: 动态相机路径下的半区敏感性测试。
// target: 1=prev(前半), 2=curr(后半), 3=both
#ifndef FLOW_DBG_PERTURB_ENABLE
#define FLOW_DBG_PERTURB_ENABLE 0
#endif
#ifndef FLOW_DBG_PERTURB_TARGET
#define FLOW_DBG_PERTURB_TARGET 1
#endif
// 交替扰动：仅在奇数 loop 注入，偶数 loop 作为同场景对照。
#ifndef FLOW_DBG_PERTURB_ALT_EVERY_OTHER
#define FLOW_DBG_PERTURB_ALT_EVERY_OTHER 1
#endif
#ifndef FLOW_DBG_PERTURB_STRIDE
#define FLOW_DBG_PERTURB_STRIDE 97
#endif
#ifndef FLOW_DBG_PERTURB_DELTA
#define FLOW_DBG_PERTURB_DELTA 31
#endif
#ifndef FLOW_DBG_PERTURB_LOG_INTERVAL
#define FLOW_DBG_PERTURB_LOG_INTERVAL 1
#endif
// 强制 loop 统计打印频率，便于奇偶扰动对照分析。
#ifndef FLOW_DBG_LOOP_LOG_INTERVAL
#define FLOW_DBG_LOOP_LOG_INTERVAL 20  // 减少日志输出频率
#endif
// R15-P2: 输出中心块原始 int8，便于与离线脚本直接对比。
#ifndef FLOW_DBG_DUMP_CENTER_PATCH
#define FLOW_DBG_DUMP_CENTER_PATCH 0  // 关闭调试输出
#endif
#ifndef FLOW_DBG_DUMP_INTERVAL
#define FLOW_DBG_DUMP_INTERVAL 40
#endif
#ifndef FLOW_DBG_PATCH_W
#define FLOW_DBG_PATCH_W 16
#endif
#ifndef FLOW_DBG_PATCH_H
#define FLOW_DBG_PATCH_H 16
#endif
// D10: CPU 模式开关（2026-02-24）
// 设置为 1 时使用纯 CPU 推理（需要 non-vela 模型）
// 设置为 0 时使用 NPU 推理（需要 vela 编译后的模型）
#ifndef FLOW_USE_CPU_INFERENCE
#define FLOW_USE_CPU_INFERENCE 0  // 恢复 NPU 模式
#endif
// D14: 彩色光流输出开关（2026-02-24）
// 设置为 1 时输出彩色光流（颜色=方向，亮度=幅度）
// 设置为 0 时输出灰度光流（亮度=幅度）
// 注意：RGB JPEG 可能超出 24KB buffer，如果失败会回退到灰度
#define FLOW_VIZ_RGB_OUTPUT 1  // R4: 恢复彩色光流输出
// D6: 诊断开关。开启后，INVOKE.image 直接发布 NPU 输入 prev 帧（灰度），用于隔离预处理链路问题。
#ifndef FLOW_DBG_VIZ_INPUT_PREV
#define FLOW_DBG_VIZ_INPUT_PREV 0  // D6: 开启为 1 可视化 NPU 输入 prev 帧
#endif
static int8_t *g_freeze_prev_q = nullptr;
static bool g_freeze_pair_ready = false;
static uint32_t g_freeze_pair_loop = 0U;
#if FLOW_DBG_OFFLINE_INJECT
static bool g_offline_pair_ready = false;
static FATFS g_offline_fs = {};
static bool g_offline_fs_mounted = false;
#endif
#if FLOW_DBG_SYNTH_INJECT
static bool g_synth_pair_ready = false;
#endif
// 4X 子采样下的 160x120 JPEG 可能小于 1KB，阈值需放宽。
constexpr uint32_t kMinJpegBytes = 128U;
constexpr uint32_t kMaxJpegBytes = 256U * 1024U;
constexpr uint32_t kJpegScanLen = 0x4B000U;
constexpr uint16_t kFallbackJpegW = 64U;
constexpr uint16_t kFallbackJpegH = 48U;
// 光流渲染缓冲：176x224 灰度 + 24KB JPEG
// D14: 分块渲染策略 - RGB buffer 动态分配
constexpr int kFlowVizMaxPixels = 176 * 224;
constexpr int kFlowVizRgbBlockRows = 8;  // 分块：每次处理 8 行
constexpr size_t kFlowVizRgbBlockSize = kFlowVizRgbBlockRows * 224 * 3;  // 8 * 224 * 3 = 5,376 bytes
constexpr size_t kFlowVizGrayJpegBufSize = 24576U;  // 24KB (灰度模式)
constexpr size_t kFlowVizRgbJpegBufSize = 49152U;   // 48KB (RGB 模式)
__attribute__((section(".bss.NoInit"))) static uint8_t g_flow_viz_gray[kFlowVizMaxPixels] __attribute__((aligned(32)));
__attribute__((section(".bss.NoInit"))) static uint8_t g_flow_viz_jpeg[kFlowVizGrayJpegBufSize] __attribute__((aligned(32)));
// D14: RGB block buffer - 静态分配（仅 5KB）
__attribute__((section(".bss.NoInit"))) static uint8_t g_flow_viz_rgb_block[kFlowVizRgbBlockSize] __attribute__((aligned(32)));
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

// 将 prev_q (H*W*3) 和 curr_q (H*W*3) 交错写入 dst_6ch (H*W*6)
// NHWC 布局：每像素 [prev_R, prev_G, prev_B, curr_R, curr_G, curr_B]
static void interleave_prev_curr_nhwc(int8_t *dst_6ch,
                                       const int8_t *prev_q,
                                       const int8_t *curr_q,
                                       size_t pix_cnt)
{
    for (size_t i = 0; i < pix_cnt; ++i) {
        const size_t s3 = i * 3U;
        const size_t d6 = i * 6U;
        dst_6ch[d6 + 0] = prev_q[s3 + 0];
        dst_6ch[d6 + 1] = prev_q[s3 + 1];
        dst_6ch[d6 + 2] = prev_q[s3 + 2];
        dst_6ch[d6 + 3] = curr_q[s3 + 0];
        dst_6ch[d6 + 4] = curr_q[s3 + 1];
        dst_6ch[d6 + 5] = curr_q[s3 + 2];
    }
}

static void render_input_prev_q_to_gray(uint8_t *out_gray,
                                        const int8_t *input_prev_q,
                                        int in_w,
                                        int in_h)
{
    if (out_gray == nullptr || input_prev_q == nullptr || in_w <= 0 || in_h <= 0) {
        return;
    }
    const size_t pix_cnt = (size_t)in_w * (size_t)in_h;
    for (size_t i = 0; i < pix_cnt; ++i) {
        const size_t off = i * 3U;
        const int r = (int)input_prev_q[off + 0] + 128;
        const int g = (int)input_prev_q[off + 1] + 128;
        const int b = (int)input_prev_q[off + 2] + 128;
        const int y = (77 * r + 150 * g + 29 * b) >> 8;
        out_gray[i] = (uint8_t)((y < 0) ? 0 : (y > 255 ? 255 : y));
    }
}

#if FLOW_DBG_PERTURB_ENABLE
static bool perturb_should_apply_loop(int loop_cnt)
{
#if FLOW_DBG_PERTURB_ALT_EVERY_OTHER
    return (loop_cnt & 1) != 0;
#else
    (void)loop_cnt;
    return true;
#endif
}

static void perturb_half_inplace(int8_t *buf_q, size_t len, int loop_cnt, const char *tag)
{
    if (buf_q == nullptr || len == 0U || tag == nullptr) {
        return;
    }
    size_t touched = 0U;
    for (size_t i = 0; i < len; i += (size_t)FLOW_DBG_PERTURB_STRIDE) {
        int v = (int)buf_q[i] + FLOW_DBG_PERTURB_DELTA;
        if (v > 127) {
            v = 127;
        } else if (v < -128) {
            v = -128;
        }
        buf_q[i] = (int8_t)v;
        ++touched;
    }
    if (ob_should_log(loop_cnt, FLOW_DBG_PERTURB_LOG_INTERVAL)) {
        xprintf("[perturb_%s] loop=%d delta=%d stride=%d touched=%u/%u\n",
                tag,
                loop_cnt,
                FLOW_DBG_PERTURB_DELTA,
                FLOW_DBG_PERTURB_STRIDE,
                (unsigned int)touched,
                (unsigned int)len);
    }
}
#endif


#if FLOW_DBG_SYNTH_INJECT
static int clamp_int(int v, int lo, int hi)
{
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static int8_t synth_q_at(int x, int y, int c)
{
    const uint32_t xu = (uint32_t)x;
    const uint32_t yu = (uint32_t)y;
    const uint32_t cu = (uint32_t)c;
    const uint32_t raw =
        (xu * 17U + yu * 29U + (xu * yu * 3U) + ((xu ^ yu) * 11U) + cu * 53U) & 0xFFU;
    return (int8_t)((int)raw - 128);
}

static bool build_synth_q_pair(int8_t *prev_q, int8_t *curr_q, int w, int h)
{
    if (prev_q == nullptr || curr_q == nullptr || w <= 0 || h <= 0) {
        return false;
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int sx = clamp_int(x - FLOW_DBG_SYNTH_SHIFT_X, 0, w - 1);
            const int sy = clamp_int(y - FLOW_DBG_SYNTH_SHIFT_Y, 0, h - 1);
            const size_t pix = (size_t)y * (size_t)w + (size_t)x;
            const size_t off = pix * 3U;
            for (int c = 0; c < 3; ++c) {
#if FLOW_DBG_SYNTH_PREV_CONST
                prev_q[off + (size_t)c] = 0;
#else
                prev_q[off + (size_t)c] = synth_q_at(x, y, c);
#endif
#if FLOW_DBG_SYNTH_CURR_CONST
                curr_q[off + (size_t)c] = 0;
#else
                curr_q[off + (size_t)c] = synth_q_at(sx, sy, c);
#endif
            }
        }
    }
    return true;
}
#endif

#if FLOW_DBG_OFFLINE_INJECT
static int load_q_file_from_sd(const char *path, int8_t *dst, size_t bytes)
{
    if (path == nullptr || dst == nullptr || bytes == 0U) {
        return -1;
    }
    if (!g_offline_fs_mounted) {
        if (f_mount(&g_offline_fs, "0:", 1) != FR_OK) {
            xprintf("[offline_inject] SD mount fail\n");
            return -1;
        }
        g_offline_fs_mounted = true;
    }

    FIL fp = {};
    UINT br = 0U;
    if (f_open(&fp, path, FA_READ) != FR_OK) {
        xprintf("[offline_inject] open fail: %s\n", path);
        return -1;
    }
    const FRESULT rr = f_read(&fp, dst, bytes, &br);
    f_close(&fp);
    if (rr != FR_OK || br != bytes) {
        xprintf("[offline_inject] read fail: %s br=%u exp=%u\n",
                path,
                (unsigned int)br,
                (unsigned int)bytes);
        return -1;
    }
    return 0;
}

static bool try_load_offline_q_pair(int8_t *prev_q, int8_t *curr_q, size_t bytes_per_frame)
{
    if (prev_q == nullptr || curr_q == nullptr || bytes_per_frame == 0U) {
        return false;
    }
    if (load_q_file_from_sd(FLOW_DBG_OFFLINE_PREV_Q_PATH, prev_q, bytes_per_frame) != 0) {
        return false;
    }
    if (load_q_file_from_sd(FLOW_DBG_OFFLINE_CURR_Q_PATH, curr_q, bytes_per_frame) != 0) {
        return false;
    }
    xprintf("[offline_inject] loaded prev=%s curr=%s bytes=%u\n",
            FLOW_DBG_OFFLINE_PREV_Q_PATH,
            FLOW_DBG_OFFLINE_CURR_Q_PATH,
            (unsigned int)bytes_per_frame);
    return true;
}
#endif

static void log_center_patch_input_q(const int8_t *prev_q,
                                     const int8_t *curr_q,
                                     int in_w,
                                     int in_h,
                                     int loop_cnt)
{
#if FLOW_DBG_DUMP_CENTER_PATCH
    static_assert(FLOW_DBG_PATCH_W > 0 && FLOW_DBG_PATCH_H > 0, "patch size must be positive");
    if (prev_q == nullptr || curr_q == nullptr || in_w <= 0 || in_h <= 0) {
        return;
    }
    const int cx = in_w / 2;
    const int cy = in_h / 2;
    const int start_x = cx - (FLOW_DBG_PATCH_W / 2);
    const int start_y = cy - (FLOW_DBG_PATCH_H / 2);
    xprintf("[dump_in_center_q] loop=%d c=(%d,%d) wh=(%d,%d)\n",
            loop_cnt,
            cx,
            cy,
            FLOW_DBG_PATCH_W,
            FLOW_DBG_PATCH_H);
    for (int iy = 0; iy < FLOW_DBG_PATCH_H; ++iy) {
        const int y = start_y + iy;
        xprintf("  in_row y=%d :", y);
        for (int ix = 0; ix < FLOW_DBG_PATCH_W; ++ix) {
            const int x = start_x + ix;
            if (x < 0 || y < 0 || x >= in_w || y >= in_h) {
                xprintf(" --");
                continue;
            }
            const int idx = (y * in_w + x) * 3;
            const int p0 = (int)prev_q[idx + 0];
            const int p1 = (int)prev_q[idx + 1];
            const int p2 = (int)prev_q[idx + 2];
            const int c0 = (int)curr_q[idx + 0];
            const int c1 = (int)curr_q[idx + 1];
            const int c2 = (int)curr_q[idx + 2];
            xprintf(" p[%d,%d,%d]/c[%d,%d,%d]", p0, p1, p2, c0, c1, c2);
        }
        xprintf("\n");
    }
#else
    (void)prev_q;
    (void)curr_q;
    (void)in_w;
    (void)in_h;
    (void)loop_cnt;
#endif
}

static void log_center_patch_out_q(const int8_t *out_q,
                                   int out_w,
                                   int out_h,
                                   int out_stride,
                                   int loop_cnt)
{
#if FLOW_DBG_DUMP_CENTER_PATCH
    static_assert(FLOW_DBG_PATCH_W > 0 && FLOW_DBG_PATCH_H > 0, "patch size must be positive");
    if (out_q == nullptr || out_w <= 0 || out_h <= 0 || out_stride < 2) {
        return;
    }
    const int cx = out_w / 2;
    const int cy = out_h / 2;
    const int start_x = cx - (FLOW_DBG_PATCH_W / 2);
    const int start_y = cy - (FLOW_DBG_PATCH_H / 2);
    xprintf("[dump_out_center_q] loop=%d c=(%d,%d) wh=(%d,%d)\n",
            loop_cnt,
            cx,
            cy,
            FLOW_DBG_PATCH_W,
            FLOW_DBG_PATCH_H);
    for (int iy = 0; iy < FLOW_DBG_PATCH_H; ++iy) {
        const int y = start_y + iy;
        xprintf("  out_row y=%d :", y);
        for (int ix = 0; ix < FLOW_DBG_PATCH_W; ++ix) {
            const int x = start_x + ix;
            if (x < 0 || y < 0 || x >= out_w || y >= out_h) {
                xprintf(" --");
                continue;
            }
            const int idx = (y * out_w + x) * out_stride;
            xprintf(" [%d,%d]", (int)out_q[idx + 0], (int)out_q[idx + 1]);
        }
        xprintf("\n");
    }
#else
    (void)out_q;
    (void)out_w;
    (void)out_h;
    (void)out_stride;
    (void)loop_cnt;
#endif
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
                               float flow_scale,
                               const int8_t *input_prev_q,
                               int input_w,
                               int input_h)
{
    viz_uart_poll_host_cmd();
    const uint8_t transport_mode = viz_uart_get_transport_mode();
    const bool send_uart = (transport_mode == 0U || transport_mode == 2U);
    const bool send_spi = (transport_mode == 1U || transport_mode == 2U);
    const bool need_uart_invoke = send_uart;
    const uint32_t algo_tick_cycles =
        (total_us <= (UINT32_MAX / 400U)) ? (total_us * 400U) : UINT32_MAX;

#if FLOW_DBG_VIZ_INPUT_PREV
    // D6 隔离实验：优先发布 NPU 输入 prev 帧，确认预处理/搬运链路是否已损坏。
    if (input_prev_q != nullptr &&
        input_w > 0 && input_h > 0 &&
        input_w * input_h <= kFlowVizMaxPixels) {
        render_input_prev_q_to_gray(g_flow_viz_gray, input_prev_q, input_w, input_h);
        hx_CleanDCache_by_Addr((volatile void *)g_flow_viz_gray, (size_t)input_w * (size_t)input_h);
        const size_t jpeg_sz = flow_render_gray_to_jpeg(g_flow_viz_gray,
                                                        input_w,
                                                        input_h,
                                                        g_flow_viz_jpeg,
                                                        kFlowVizGrayJpegBufSize);
        if (jpeg_sz > 0U) {
            hx_CleanDCache_by_Addr((volatile void *)g_flow_viz_jpeg, jpeg_sz);
            if (send_uart) {
                viz_uart_send_device_id_once();
                if (need_uart_invoke) {
                    viz_uart_send_invoke_jpeg(g_flow_viz_jpeg,
                                              jpeg_sz,
                                              (uint16_t)input_w,
                                              (uint16_t)input_h,
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

    // 阶段 D：有 flow 输出时优先发送光流渲染图 (灰度)
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
        size_t jpeg_sz = 0U;
#if FLOW_VIZ_RGB_OUTPUT
        // D14/R3: 分块彩色光流输出 - 颜色=方向，亮度=幅度
        // RGB JPEG 可能超出 24KB，失败时回退到灰度模式
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
            // RGB 编码失败，回退到灰度模式
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
    g_ctx.log_print_interval = FLOW_DBG_LOOP_LOG_INTERVAL;

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

#if FLOW_USE_CPU_INFERENCE
        // D10: CPU 模式 - 使用纯 CPU 算子运行 non-vela 模型
        // 需要添加：CONV_2D, ADD, TRANSPOSE_CONV, RESIZE_BILINEAR, STRIDED_SLICE
        static tflite::MicroMutableOpResolver<6> yolov8n_ob_op_resolver;
        yolov8n_ob_op_resolver.AddConv2D();
        yolov8n_ob_op_resolver.AddAdd();
        yolov8n_ob_op_resolver.AddTransposeConv();
        yolov8n_ob_op_resolver.AddResizeBilinear();
        yolov8n_ob_op_resolver.AddStridedSlice();
        yolov8n_ob_op_resolver.AddTranspose();
        xprintf("[CPU_MODE] Using pure CPU inference with non-vela model\n");
#else
        // NPU 模式 - 使用 Ethos-U 算子运行 vela 编译后的模型
        static tflite::MicroMutableOpResolver<2> yolov8n_ob_op_resolver;

        yolov8n_ob_op_resolver.AddTranspose();
        if (kTfLiteOk != yolov8n_ob_op_resolver.AddEthosU()) {
            xprintf("Failed to add Arm NPU support to op resolver.");
            return false;
        }
        xprintf("[NPU_MODE] Using Ethos-U NPU inference with vela model\n");
#endif

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

        const int output_cnt = yolov8n_ob_static_interpreter.outputs_size();
        xprintf("model outputs=%d\r\n", output_cnt);
        for (int oi = 0; oi < output_cnt; ++oi) {
            TfLiteTensor *out_i = yolov8n_ob_static_interpreter.output(oi);
            if (out_i == nullptr || out_i->dims == nullptr) {
                xprintf("[out_tensor=%d] metadata missing\r\n", oi);
                continue;
            }
            const int scale_1e6 = (int)(out_i->params.scale * 1000000.0f);
            xprintf("[out_tensor=%d] type=%d dims=[", oi, out_i->type);
            for (int di = 0; di < out_i->dims->size; ++di) {
                if (di > 0) {
                    xprintf(",");
                }
                xprintf("%d", out_i->dims->data[di]);
            }
            xprintf("] scale=%d.%06d zp=%d\r\n",
                    scale_1e6 / 1000000,
                    abs(scale_1e6 % 1000000),
                    out_i->params.zero_point);
        }

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
    const uint32_t curr_shadow_addr = mm_reserve_align((uint32_t)g_raw_frame_bytes, 0x20);
    if (curr_shadow_addr == 0U) {
        xprintf("alloc curr shadow fail, size=%u\r\n", (unsigned int)g_raw_frame_bytes);
        return -1;
    }
    g_curr_q_shadow = (int8_t *)curr_shadow_addr;

    // prev_q 缓冲区：存储上一帧，用于与当前帧交错拼装成 6 通道 NHWC 输入
    const uint32_t prev_buffer_addr = mm_reserve_align((uint32_t)g_raw_frame_bytes, 0x20);
    if (prev_buffer_addr == 0U) {
        xprintf("alloc prev buffer fail, size=%u\r\n", (unsigned int)g_raw_frame_bytes);
        return -1;
    }
    g_prev_q_buffer = (int8_t *)prev_buffer_addr;
    xprintf("prev_q buffer allocated at 0x%x size=%u\n", prev_buffer_addr, (unsigned int)g_raw_frame_bytes);

#if FLOW_DBG_FREEZE_PAIR
    const uint32_t freeze_prev_addr = mm_reserve_align((uint32_t)g_raw_frame_bytes, 0x20);
    if (freeze_prev_addr != 0U) {
        g_freeze_prev_q = (int8_t *)freeze_prev_addr;
        g_freeze_pair_ready = false;
        g_freeze_pair_loop = 0U;
        xprintf("[freeze_pair] enabled bytes=%u (reuse curr_shadow)\n", (unsigned int)g_raw_frame_bytes);
    } else {
        g_freeze_prev_q = nullptr;
        g_freeze_pair_ready = false;
        xprintf("[freeze_pair] disabled due to alloc fail\n");
    }
#endif
#if FLOW_DBG_OFFLINE_INJECT
    if (g_freeze_prev_q != nullptr && g_curr_q_shadow != nullptr) {
        if (try_load_offline_q_pair(g_freeze_prev_q, g_curr_q_shadow, g_raw_frame_bytes)) {
            g_offline_pair_ready = true;
            g_freeze_pair_ready = true;
            g_freeze_pair_loop = 0U;
            g_prev_frame_valid = true;
            xprintf("[offline_inject] enabled; camera input bypassed\n");
        } else {
            g_offline_pair_ready = false;
            xprintf("[offline_inject] disabled; fallback to camera input\n");
        }
    } else {
        xprintf("[offline_inject] disabled due to missing freeze buffers\n");
    }
#endif
#if FLOW_DBG_SYNTH_INJECT
    if (g_freeze_prev_q != nullptr && g_curr_q_shadow != nullptr) {
#if FLOW_DBG_OFFLINE_INJECT
        if (!g_offline_pair_ready)
#endif
        {
            if (build_synth_q_pair(g_freeze_prev_q, g_curr_q_shadow, g_model_in_w, g_model_in_h)) {
                g_synth_pair_ready = true;
                g_freeze_pair_ready = true;
                g_freeze_pair_loop = 0U;
                g_prev_frame_valid = true;
                xprintf("[synth_inject] enabled shift=(%d,%d); camera input bypassed\n",
                        FLOW_DBG_SYNTH_SHIFT_X,
                        FLOW_DBG_SYNTH_SHIFT_Y);
            } else {
                g_synth_pair_ready = false;
                xprintf("[synth_inject] disabled due to build fail\n");
            }
        }
    } else {
        xprintf("[synth_inject] disabled due to missing freeze buffers\n");
    }
#endif

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
    if (g_curr_q_shadow == nullptr) {
        xprintf("curr shadow buffer not allocated\r\n");
        return -1;
    }
    ob_perf_stamp_t t_total_start;
    ob_perf_stamp_t t_total_end;
    ob_perf_stamp_t t_io_start;
    ob_perf_stamp_t t_io_end;
    ob_perf_stamp_t t_preproc_start;
    ob_perf_stamp_t t_preproc_end;
    ob_perf_stamp_t t_infer_start;
    ob_perf_stamp_t t_infer_end;
    ob_checksum_stats_t curr_q_before = {};
    ob_checksum_stats_t curr_q_after = {};

    ob_perf_mark(&t_total_start);
    ob_perf_mark(&t_io_start);

    const size_t pix_cnt = (size_t)g_model_in_w * (size_t)g_model_in_h;
    int8_t *input_ptr = (int8_t *)yolov8n_ob_input->data.data;
    // 当前帧始终写入独立 shadow，避免与 NHWC 目标 input_ptr 重叠。
    int8_t *curr_q = g_curr_q_shadow;
    if (curr_q == nullptr) {
        xprintf("curr shadow buffer not allocated\r\n");
        return -1;
    }
    uint8_t *curr_raw = reinterpret_cast<uint8_t *>(curr_q);

    bool injected_pair_ready = false;
#if FLOW_DBG_OFFLINE_INJECT
    injected_pair_ready = g_offline_pair_ready;
#endif
#if FLOW_DBG_SYNTH_INJECT
    injected_pair_ready = injected_pair_ready || g_synth_pair_ready;
#endif

    if (injected_pair_ready && g_freeze_prev_q != nullptr) {
        memcpy(input_ptr, g_freeze_prev_q, g_raw_frame_bytes);
        if (curr_q != g_curr_q_shadow) {
            memcpy(curr_q, g_curr_q_shadow, g_raw_frame_bytes);
        }
        compute_checksum_from_q(input_ptr, g_raw_frame_bytes, &g_ctx.raw1_stats);
        compute_checksum_from_q(curr_q, g_raw_frame_bytes, &g_ctx.raw2_stats);
    } else {
        if (cam_input_get_frame(curr_raw, g_raw_frame_bytes) != 0) {
            xprintf("camera frame capture fail\n");
            return -1;
        }
        ob_compute_checksum(curr_raw, g_raw_frame_bytes, &g_ctx.raw2_stats);
        quantize_rgb_frame_inplace(curr_raw, curr_q, g_raw_frame_bytes);

        if (!g_prev_frame_valid) {
            // 首帧：存入 prev 缓冲区，等待下一帧配对
            memcpy(g_prev_q_buffer, curr_q, g_raw_frame_bytes);
            g_prev_frame_valid = true;
            if (g_ctx.loop_cnt < 3) {
                xprintf("[NHWC] first frame stored to prev_q_buffer\n");
            }
            return 0;
        }
        compute_checksum_from_q(g_prev_q_buffer, g_raw_frame_bytes, &g_ctx.raw1_stats);
    }
    ob_perf_mark(&t_io_end);
    ob_perf_mark(&t_preproc_start);

    // 核心修复：将 prev (g_prev_q_buffer) 和 curr (curr_q) 交错拼装到 6 通道 NHWC 输入
    interleave_prev_curr_nhwc(input_ptr, g_prev_q_buffer, curr_q, pix_cnt);
    if (g_ctx.loop_cnt < 3) {
        xprintf("[NHWC] interleaved prev+curr into input tensor (%u pixels)\n",
                (unsigned int)pix_cnt);
        // 打印前 12 字节验证交错正确性
        xprintf("[NHWC] input[0..11]: %d %d %d %d %d %d | %d %d %d %d %d %d\n",
                (int)input_ptr[0], (int)input_ptr[1], (int)input_ptr[2],
                (int)input_ptr[3], (int)input_ptr[4], (int)input_ptr[5],
                (int)input_ptr[6], (int)input_ptr[7], (int)input_ptr[8],
                (int)input_ptr[9], (int)input_ptr[10], (int)input_ptr[11]);
    }

    compute_checksum_from_q(g_prev_q_buffer, g_raw_frame_bytes, &g_ctx.raw1_stats);
    compute_checksum_from_q(curr_q, g_raw_frame_bytes, &curr_q_before);
    compute_checksum_from_q(curr_q, g_raw_frame_bytes, &g_ctx.raw2_stats);

    ob_perf_mark(&t_preproc_end);
    ob_perf_mark(&t_infer_start);

    const TfLiteStatus invoke_status = yolov8n_ob_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        xprintf("optical flow invoke fail\n");
        return -1;
    }
    compute_checksum_from_q(curr_q, g_raw_frame_bytes, &curr_q_after);
    // Invoke 后：当前帧变成下一帧的 prev
    memcpy(g_prev_q_buffer, curr_q, g_raw_frame_bytes);
#if FLOW_DBG_FREEZE_PAIR
    if (g_freeze_pair_ready && g_freeze_prev_q != nullptr) {
        memcpy(input_ptr, g_freeze_prev_q, g_raw_frame_bytes);
    }
#endif

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
    static bool s_quant_logged = false;
    if (!s_quant_logged) {
        const int in_scale_1e6 = (int)(yolov8n_ob_input->params.scale * 1000000.0f);
        const int out_scale_1e6 = (int)(out_scale * 1000000.0f);
        xprintf("[quant] in: type=%d scale=%d.%06d zp=%d | out: type=%d scale=%d.%06d zp=%d\r\n",
                yolov8n_ob_input->type,
                in_scale_1e6 / 1000000,
                abs(in_scale_1e6 % 1000000),
                yolov8n_ob_input->params.zero_point,
                yolov8n_ob_output->type,
                out_scale_1e6 / 1000000,
                abs(out_scale_1e6 % 1000000),
                out_zp);
        s_quant_logged = true;
    }

    publish_viz_payload(algoresult_yolov8n_ob,
                       g_ctx.total_us,
                       g_ctx.loop_cnt,
                       out_data,
                       g_model_out_w,
                       g_model_out_h,
                       g_model_out_c,
                       out_zp,
                       out_scale,
                       g_prev_q_buffer,
                       g_model_in_w,
                       g_model_in_h);

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
        ob_log_mag_stats_grid_sample(out_data,
                                     g_model_out_w,
                                     g_model_out_h,
                                     g_model_out_c,
                                     out_zp,
                                     out_scale);
        ob_log_out_q_histogram(out_data,
                               g_model_out_w,
                               g_model_out_h,
                               g_model_out_c);
        xprintf("[curr_q pre/post] pre sum=%u min=%u max=%u | post sum=%u min=%u max=%u\r\n",
                curr_q_before.sum,
                curr_q_before.min,
                curr_q_before.max,
                curr_q_after.sum,
                curr_q_after.min,
                curr_q_after.max);
#if FLOW_DBG_FREEZE_PAIR
        if (g_freeze_pair_ready) {
            xprintf("[freeze_pair] active captured_loop=%u\n", g_freeze_pair_loop);
        }
#endif
    }

#if FLOW_DBG_DUMP_CENTER_PATCH
    if (ob_should_log(g_ctx.loop_cnt, FLOW_DBG_DUMP_INTERVAL)) {
        log_center_patch_input_q(input_ptr, g_curr_q_shadow, g_model_in_w, g_model_in_h, g_ctx.loop_cnt);
        log_center_patch_out_q(out_data, g_model_out_w, g_model_out_h, g_model_out_c, g_ctx.loop_cnt);
    }
#endif

    g_ctx.loop_cnt++;
    return ercode;
}

int cv_yolov8n_ob_deinit()
{
    cam_input_deinit();
    return 0;
}

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
#include "cvapp_optical_flow.h"
#include "ethosu_driver.h"
#include "hx_drv_gpio.h"
#include "hx_drv_jpeg.h"
#include "hx_drv_scu.h"
#include "cam_input.h"
#include "memory_manage.h"
#include "ob_debug_stats.h"
#include "ob_perf.h"
#include "ob_runtime_ctx.h"
#include "flow_render.h"
#include "frame_quantize.h"
#include "viz_publish.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "xprintf.h"
#include "ff.h"

#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif

#include "common_config.h"

// Shared optical-flow model parameters live in common_config.h.
#define INPUT_IMAGE_CHANNELS FLOW_MODEL_CHANNELS
#define OPTICAL_FLOW_INPUT_TENSOR_CHANNEL FLOW_MODEL_CHANNELS

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

static uint32_t tensor_arena = 0;
static uint32_t g_tensor_arena_size = 0U;
static ethosu_driver ethosu_drv;

static tflite::MicroInterpreter *optical_flow_int_ptr = nullptr;
static TfLiteTensor *optical_flow_input = nullptr;
static TfLiteTensor *optical_flow_output = nullptr;
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
static bool g_prev_frame_valid = false;

static ob_runtime_ctx_t g_ctx = {};

static bool load_model_io_metadata(const tflite::Model *model)
{
    if (model == nullptr || model->subgraphs() == nullptr || model->subgraphs()->size() <= 0) {
        xprintf("model subgraph metadata missing\n");
        return false;
    }

    const auto *subgraph = model->subgraphs()->Get(0);
    if (subgraph == nullptr || subgraph->tensors() == nullptr ||
        subgraph->inputs() == nullptr || subgraph->outputs() == nullptr ||
        subgraph->inputs()->size() <= 0 || subgraph->outputs()->size() <= 0) {
        xprintf("model tensor metadata missing\n");
        return false;
    }

    const int input_idx = subgraph->inputs()->Get(0);
    const int output_idx = subgraph->outputs()->Get(0);
    const auto *input_tensor = subgraph->tensors()->Get(input_idx);
    const auto *output_tensor = subgraph->tensors()->Get(output_idx);
    if (input_tensor == nullptr || output_tensor == nullptr ||
        input_tensor->shape() == nullptr || output_tensor->shape() == nullptr) {
        xprintf("model shape metadata missing\n");
        return false;
    }

    const auto *in_shape = input_tensor->shape();
    if (in_shape->size() != 4) {
        xprintf("input dims size unsupported: %d\n", in_shape->size());
        return false;
    }

    g_model_in_h = in_shape->Get(1);
    g_model_in_w = in_shape->Get(2);
    g_model_in_c = in_shape->Get(3);
    if (g_model_in_w <= 0 || g_model_in_h <= 0 || g_model_in_c != INPUT_IMAGE_CHANNELS) {
        xprintf("input dims invalid: h=%d w=%d c=%d expect_c=%d\n",
                g_model_in_h,
                g_model_in_w,
                g_model_in_c,
                INPUT_IMAGE_CHANNELS);
        return false;
    }

    const auto *out_shape = output_tensor->shape();
    if (out_shape->size() == 4) {
        g_model_out_h = out_shape->Get(1);
        g_model_out_w = out_shape->Get(2);
        g_model_out_c = out_shape->Get(3);
    } else if (out_shape->size() == 3) {
        g_model_out_h = out_shape->Get(0);
        g_model_out_w = out_shape->Get(1);
        g_model_out_c = out_shape->Get(2);
    } else {
        xprintf("output dims size unsupported: %d\n", out_shape->size());
        return false;
    }

    if (g_model_out_w <= 0 || g_model_out_h <= 0 || g_model_out_c < 2) {
        xprintf("output dims invalid: h=%d w=%d c=%d\n",
                g_model_out_h,
                g_model_out_w,
                g_model_out_c);
        return false;
    }

    g_raw_frame_bytes = (size_t)g_model_in_w * (size_t)g_model_in_h * 3U;
    return true;
}

static uint32_t compute_dynamic_arena_budget(void)
{
    const uint32_t remaining = mm_get_remaining();
    if (remaining <= FLOW_TENSOR_ARENA_SAFETY_BYTES) {
        return 0U;
    }
    const uint32_t usable = remaining - FLOW_TENSOR_ARENA_SAFETY_BYTES;
    return usable & ~0x1FU;
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





}  // namespace

int cv_optical_flow_init(bool security_enable, bool privilege_enable, uint32_t model_addr)
{
    int ercode = 0;
    const tflite::Model *optical_flow_model = nullptr;
    uint32_t mm_before_frame_buffers = 0U;
    uint32_t mm_after_frame_buffers = 0U;
    uint32_t mm_after_sensor_buffers = 0U;
    uint32_t mm_after_viz_buffers = 0U;
    uint32_t frame_buffers_used = 0U;
    uint32_t sensor_buffers_used = 0U;
    uint32_t viz_buffers_used = 0U;
    uint32_t remaining_before_arena = 0U;
    uint32_t remaining_after_arena = 0U;

    g_ctx.loop_cnt = 0;
#ifdef VIZ_UART_MODE
    // VIZ 模式下保留低频性能与亮度统计，支撑阶段 A 调优。
    g_ctx.log_print_interval = 20;
#else
    g_ctx.log_print_interval = 5;
#endif
    g_ctx.log_print_interval = 5;

    if (model_addr != 0) {
        optical_flow_model = tflite::GetModel((const void *)model_addr);

        if (optical_flow_model->version() != TFLITE_SCHEMA_VERSION) {
            xprintf("[ERROR] model schema %d != %d\n",
                    optical_flow_model->version(),
                    TFLITE_SCHEMA_VERSION);
            return -1;
        }
        xprintf("model schema %d\n", optical_flow_model->version());

        if (!load_model_io_metadata(optical_flow_model)) {
            return -1;
        }
        xprintf("model io: in(h=%d,w=%d,c=%d) out(h=%d,w=%d,c=%d)\n",
                g_model_in_h,
                g_model_in_w,
                g_model_in_c,
                g_model_out_h,
                g_model_out_w,
                g_model_out_c);

        xprintf("[mem] mm remaining at model parse=%u\n",
                (unsigned int)mm_get_remaining());
    }

    g_prev_frame_valid = false;
    mm_before_frame_buffers = mm_get_current_addr();

    const uint32_t curr_shadow_addr = mm_reserve_align((uint32_t)g_raw_frame_bytes, 0x20);
    if (curr_shadow_addr == 0U) {
        xprintf("alloc curr shadow fail, size=%u\r\n", (unsigned int)g_raw_frame_bytes);
        return -1;
    }
    g_curr_q_shadow = (int8_t *)curr_shadow_addr;

    const uint32_t prev_buffer_addr = mm_reserve_align((uint32_t)g_raw_frame_bytes, 0x20);
    if (prev_buffer_addr == 0U) {
        xprintf("alloc prev buffer fail, size=%u\r\n", (unsigned int)g_raw_frame_bytes);
        return -1;
    }
    g_prev_q_buffer = (int8_t *)prev_buffer_addr;
    xprintf("prev_q buffer allocated at 0x%x size=%u\n", prev_buffer_addr, (unsigned int)g_raw_frame_bytes);
    mm_after_frame_buffers = mm_get_current_addr();
    frame_buffers_used = mm_after_frame_buffers - mm_before_frame_buffers;

    if (cam_input_init((uint32_t)g_model_in_w, (uint32_t)g_model_in_h) != 0) {
        xprintf("camera init fail\n");
        return -1;
    }
    mm_after_sensor_buffers = mm_get_current_addr();
    sensor_buffers_used = mm_after_sensor_buffers - mm_after_frame_buffers;

    if (viz_publish_init(g_model_out_w, g_model_out_h) != 0) {
        xprintf("viz init fail\n");
        return -1;
    }
    mm_after_viz_buffers = mm_get_current_addr();
    viz_buffers_used = mm_after_viz_buffers - mm_after_sensor_buffers;

    remaining_before_arena = mm_get_remaining();
    g_tensor_arena_size = compute_dynamic_arena_budget();
    if (g_tensor_arena_size == 0U) {
        xprintf("alloc tensor arena fail, remaining=%u safety=%u\n",
                (unsigned int)remaining_before_arena,
                (unsigned int)FLOW_TENSOR_ARENA_SAFETY_BYTES);
        return -1;
    }
    tensor_arena = mm_reserve_align(g_tensor_arena_size, 0x20);
    if (tensor_arena == 0) {
        xprintf("alloc tensor arena fail, size=%u\n", (unsigned int)g_tensor_arena_size);
        return -1;
    }
    remaining_after_arena = mm_get_remaining();
    xprintf("[mem] frame_buffers=%u sensor+other=%u viz_buffers=%u arena_budget=%u remaining_before_arena=%u remaining_after_arena=%u\n",
            (unsigned int)frame_buffers_used,
            (unsigned int)sensor_buffers_used,
            (unsigned int)viz_buffers_used,
            (unsigned int)g_tensor_arena_size,
            (unsigned int)remaining_before_arena,
            (unsigned int)remaining_after_arena);
    xprintf("TA[%x], size=%u\r\n", tensor_arena, (unsigned int)g_tensor_arena_size);

    if (_arm_npu_init(security_enable, privilege_enable) != 0) {
        return -1;
    }

    // 计时初始化放在启动阶段，保证 run 中只负责打点。
    ob_perf_init();

    if (model_addr != 0) {
#if TFLM2209_U55TAG2205
        static tflite::MicroErrorReporter optical_flow_micro_error_reporter;
#endif

#if FLOW_USE_CPU_INFERENCE
        // D10: CPU 模式 - 使用纯 CPU 算子运行 non-vela 模型
        // 需要添加：CONV_2D, ADD, TRANSPOSE_CONV, RESIZE_BILINEAR, STRIDED_SLICE
        static tflite::MicroMutableOpResolver<6> optical_flow_op_resolver;
        optical_flow_op_resolver.AddConv2D();
        optical_flow_op_resolver.AddAdd();
        optical_flow_op_resolver.AddTransposeConv();
        optical_flow_op_resolver.AddResizeBilinear();
        optical_flow_op_resolver.AddStridedSlice();
        optical_flow_op_resolver.AddTranspose();
        xprintf("[CPU_MODE] Using pure CPU inference with non-vela model\n");
#else
        // NPU 模式 - 使用 Ethos-U 算子运行 vela 编译后的模型
        static tflite::MicroMutableOpResolver<2> optical_flow_op_resolver;

        optical_flow_op_resolver.AddTranspose();
        if (kTfLiteOk != optical_flow_op_resolver.AddEthosU()) {
            xprintf("Failed to add Arm NPU support to op resolver.");
            return false;
        }
        xprintf("[NPU_MODE] Using Ethos-U NPU inference with vela model\n");
#endif

#if TFLM2209_U55TAG2205
        static tflite::MicroInterpreter optical_flow_static_interpreter(
            optical_flow_model,
            optical_flow_op_resolver,
            (uint8_t *)tensor_arena,
            g_tensor_arena_size,
            &optical_flow_micro_error_reporter);
#else
        static tflite::MicroInterpreter optical_flow_static_interpreter(
            optical_flow_model,
            optical_flow_op_resolver,
            (uint8_t *)tensor_arena,
            g_tensor_arena_size);
#endif

        if (optical_flow_static_interpreter.AllocateTensors() != kTfLiteOk) {
            xprintf("AllocateTensors fail, arena=%u\n", (unsigned int)g_tensor_arena_size);
            return false;
        }

        optical_flow_int_ptr = &optical_flow_static_interpreter;
        optical_flow_input = optical_flow_static_interpreter.input(0);
        optical_flow_output = optical_flow_static_interpreter.output(0);

        const int output_cnt = optical_flow_static_interpreter.outputs_size();
        xprintf("model outputs=%d\r\n", output_cnt);
        for (int oi = 0; oi < output_cnt; ++oi) {
            TfLiteTensor *out_i = optical_flow_static_interpreter.output(oi);
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

        if (optical_flow_input == nullptr || optical_flow_output == nullptr ||
            optical_flow_input->dims == nullptr || optical_flow_output->dims == nullptr) {
            xprintf("input/output tensor metadata missing\n");
            return -1;
        }

    }

    xprintf("initial done\n");
    return ercode;
}

int cv_optical_flow_run(struct_optical_flow_algoResult *algoresult_optical_flow)
{
    int ercode = 0;
    memset(algoresult_optical_flow, 0, sizeof(struct_optical_flow_algoResult));

    if (optical_flow_int_ptr == nullptr) {
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
    ob_checksum_stats_t curr_q_after = {};

    ob_perf_mark(&t_total_start);
    ob_perf_mark(&t_io_start);

    const size_t pix_cnt = (size_t)g_model_in_w * (size_t)g_model_in_h;
    int8_t *input_ptr = (int8_t *)optical_flow_input->data.data;
    // 当前帧始终写入独立 shadow，避免与 NHWC 目标 input_ptr 重叠。
    int8_t *curr_q = g_curr_q_shadow;
    if (curr_q == nullptr) {
        xprintf("curr shadow buffer not allocated\r\n");
        return -1;
    }
    uint8_t *curr_raw = reinterpret_cast<uint8_t *>(curr_q);

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
    ob_perf_mark(&t_io_end);
    ob_perf_mark(&t_preproc_start);

    // 核心修复：将 prev (g_prev_q_buffer) 和 curr (curr_q) 交错拼装到 6 通道 NHWC 输入
    interleave_prev_curr_nhwc(input_ptr, g_prev_q_buffer, curr_q, pix_cnt);
    if (g_ctx.loop_cnt < 3) {
        xprintf("[NHWC] interleaved prev+curr into input tensor (%u pixels)\n",
                (unsigned int)pix_cnt);

    }

    compute_checksum_from_q(curr_q, g_raw_frame_bytes, &g_ctx.raw2_stats);

    ob_perf_mark(&t_preproc_end);
    ob_perf_mark(&t_infer_start);

    const TfLiteStatus invoke_status = optical_flow_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        xprintf("optical flow invoke fail\n");
        return -1;
    }
    compute_checksum_from_q(curr_q, g_raw_frame_bytes, &curr_q_after);
    // Invoke 后：当前帧变成下一帧的 prev
    memcpy(g_prev_q_buffer, curr_q, g_raw_frame_bytes);


    ob_perf_mark(&t_infer_end);
    ob_perf_mark(&t_total_end);

    g_ctx.sd_us = ob_perf_elapsed_us(&t_io_start, &t_io_end);
    g_ctx.preproc_us = ob_perf_elapsed_us(&t_preproc_start, &t_preproc_end);
    g_ctx.infer_us = ob_perf_elapsed_us(&t_infer_start, &t_infer_end);
    g_ctx.total_us = ob_perf_elapsed_us(&t_total_start, &t_total_end);

    const float out_scale =
        ((TfLiteAffineQuantization *)(optical_flow_output->quantization.params))->scale->data[0];
    const int out_zp =
        ((TfLiteAffineQuantization *)(optical_flow_output->quantization.params))->zero_point->data[0];
    const int8_t *out_data = optical_flow_output->data.int8;
    static bool s_quant_logged = false;
    if (!s_quant_logged) {
        const int in_scale_1e6 = (int)(optical_flow_input->params.scale * 1000000.0f);
        const int out_scale_1e6 = (int)(out_scale * 1000000.0f);
        xprintf("[quant] in: type=%d scale=%d.%06d zp=%d | out: type=%d scale=%d.%06d zp=%d\r\n",
                optical_flow_input->type,
                in_scale_1e6 / 1000000,
                abs(in_scale_1e6 % 1000000),
                optical_flow_input->params.zero_point,
                optical_flow_output->type,
                out_scale_1e6 / 1000000,
                abs(out_scale_1e6 % 1000000),
                out_zp);
        s_quant_logged = true;
    }

    publish_viz_payload(algoresult_optical_flow,
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
        xprintf("[curr_q] sum=%u min=%u max=%u\r\n",
                g_ctx.raw2_stats.sum,
                g_ctx.raw2_stats.min,
                g_ctx.raw2_stats.max);
    }



    g_ctx.loop_cnt++;
    return ercode;
}

int cv_optical_flow_deinit()
{
    cam_input_deinit();
    return 0;
}

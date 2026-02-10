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
#include "cisdp_cfg.h"
#include "cisdp_sensor.h"
#include "cvapp_yolov8n_ob.h"
#include "ethosu_driver.h"
#include "hx_drv_gpio.h"
#include "hx_drv_scu.h"
#include "img_proc_helium.h"
#include "memory_manage.h"
#include "ob_debug_stats.h"
#include "ob_perf.h"
#include "ob_runtime_ctx.h"
#include "ob_sd_frame.h"
#include "send_result.h"
#include "spi_master_protocol.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "xprintf.h"
#include "yolo_postprocessing.h"

#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif

#define INPUT_IMAGE_CHANNELS 6
#define YOLOV8_OB_INPUT_TENSOR_WIDTH 256
#define YOLOV8_OB_INPUT_TENSOR_HEIGHT 192
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

namespace {

constexpr int tensor_arena_size = 1600 * 1024;

static uint32_t tensor_arena = 0;
static ethosu_driver ethosu_drv;

static tflite::MicroInterpreter *yolov8n_ob_int_ptr = nullptr;
static TfLiteTensor *yolov8n_ob_input = nullptr;
static TfLiteTensor *yolov8n_ob_output = nullptr;

// SD 数据集配置。
static const char *RAW_DIR = "0:/ai_master/dataset/sintel/test/clean/market_4/raw";
static const char *RAW_FMT = "frame_%04d_rgb888.raw";
static const int RAW_FRAME_BYTES =
    YOLOV8_OB_INPUT_TENSOR_WIDTH * YOLOV8_OB_INPUT_TENSOR_HEIGHT * 3;

static ob_runtime_ctx_t g_ctx = {};

static void prepare_optical_flow_input(const uint8_t *raw1,
                                       const uint8_t *raw2,
                                       int8_t *input_ptr,
                                       size_t pixel_count)
{
    for (size_t i = 0; i < pixel_count * 3U; ++i) {
        int16_t q1 = (int16_t)raw1[i] - 128;
        if (q1 > 127) {
            q1 = 127;
        }
        if (q1 < -128) {
            q1 = -128;
        }
        input_ptr[i] = (int8_t)q1;

        int16_t q2 = (int16_t)raw2[i] - 128;
        if (q2 > 127) {
            q2 = 127;
        }
        if (q2 < -128) {
            q2 = -128;
        }
        input_ptr[i + pixel_count * 3U] = (int8_t)q2;
    }
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

int cv_yolov8n_ob_init(bool security_enable, bool privilege_enable, uint32_t model_addr)
{
    int ercode = 0;

    g_ctx.raw_frame_idx = 1;
    g_ctx.raw_frame_max = 50;
    g_ctx.loop_cnt = 0;
    g_ctx.log_print_interval = 5;

    tensor_arena = mm_reserve_align(tensor_arena_size, 0x20);
    xprintf("TA[%x]\r\n", tensor_arena);

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
            return false;
        }

        yolov8n_ob_int_ptr = &yolov8n_ob_static_interpreter;
        yolov8n_ob_input = yolov8n_ob_static_interpreter.input(0);
        yolov8n_ob_output = yolov8n_ob_static_interpreter.output(0);
    }

    const uint32_t buf_base = mm_reserve_align(RAW_FRAME_BYTES * 2, 0x20);
    if (buf_base == 0) {
        xprintf("alloc raw buffer fail\n");
        return -1;
    }
    g_ctx.raw_buf_1 = (uint8_t *)buf_base;
    g_ctx.raw_buf_2 = (uint8_t *)(buf_base + RAW_FRAME_BYTES);

    if (ob_sd_init(RAW_DIR, RAW_FMT, g_ctx.raw_frame_max) != 0) {
        xprintf("sd init fail\n");
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
    if (g_ctx.raw_buf_1 == nullptr || g_ctx.raw_buf_2 == nullptr) {
        xprintf("raw buffers not allocated\n");
        return -1;
    }

    ob_perf_stamp_t t_total_start;
    ob_perf_stamp_t t_total_end;
    ob_perf_stamp_t t_sd_start;
    ob_perf_stamp_t t_sd_end;
    ob_perf_stamp_t t_preproc_start;
    ob_perf_stamp_t t_preproc_end;
    ob_perf_stamp_t t_infer_start;
    ob_perf_stamp_t t_infer_end;

    ob_perf_mark(&t_total_start);
    ob_perf_mark(&t_sd_start);

    int next_frame_idx = g_ctx.raw_frame_idx;
    if (ob_sd_load_frame_pair(g_ctx.raw_frame_idx,
                              g_ctx.raw_buf_1,
                              g_ctx.raw_buf_2,
                              RAW_FRAME_BYTES,
                              &next_frame_idx) != 0) {
        return -1;
    }

    ob_perf_mark(&t_sd_end);
    ob_perf_mark(&t_preproc_start);

    ob_compute_checksum(g_ctx.raw_buf_1, RAW_FRAME_BYTES, &g_ctx.raw1_stats);
    ob_compute_checksum(g_ctx.raw_buf_2, RAW_FRAME_BYTES, &g_ctx.raw2_stats);

    const size_t pix_cnt =
        (size_t)YOLOV8_OB_INPUT_TENSOR_WIDTH * (size_t)YOLOV8_OB_INPUT_TENSOR_HEIGHT;
    int8_t *input_ptr = (int8_t *)yolov8n_ob_input->data.data;
    prepare_optical_flow_input(g_ctx.raw_buf_1, g_ctx.raw_buf_2, input_ptr, pix_cnt);

    ob_perf_mark(&t_preproc_end);
    ob_perf_mark(&t_infer_start);

    const TfLiteStatus invoke_status = yolov8n_ob_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        xprintf("optical flow invoke fail\n");
        return -1;
    }

    ob_perf_mark(&t_infer_end);
    ob_perf_mark(&t_total_end);

    g_ctx.sd_us = ob_perf_elapsed_us(&t_sd_start, &t_sd_end);
    g_ctx.preproc_us = ob_perf_elapsed_us(&t_preproc_start, &t_preproc_end);
    g_ctx.infer_us = ob_perf_elapsed_us(&t_infer_start, &t_infer_end);
    g_ctx.total_us = ob_perf_elapsed_us(&t_total_start, &t_total_end);

    const float out_scale =
        ((TfLiteAffineQuantization *)(yolov8n_ob_output->quantization.params))->scale->data[0];
    const int out_zp =
        ((TfLiteAffineQuantization *)(yolov8n_ob_output->quantization.params))->zero_point->data[0];
    const int8_t *out_data = yolov8n_ob_output->data.int8;

    ob_flow_summary_t flow_summary = {};
    ob_compute_flow_summary(out_data,
                            YOLOV8_OB_INPUT_TENSOR_WIDTH,
                            YOLOV8_OB_INPUT_TENSOR_HEIGHT,
                            out_zp,
                            out_scale,
                            &flow_summary);

    if (ob_should_log(g_ctx.loop_cnt, g_ctx.log_print_interval)) {
        ob_log_infer_line(g_ctx.loop_cnt,
                          g_ctx.raw_frame_idx,
                          g_ctx.raw_frame_max,
                          &flow_summary,
                          &g_ctx.raw1_stats,
                          &g_ctx.raw2_stats,
                          g_ctx.sd_us,
                          g_ctx.preproc_us,
                          g_ctx.infer_us,
                          g_ctx.total_us);
    }

    g_ctx.raw_frame_idx = next_frame_idx;
    g_ctx.loop_cnt++;
    return ercode;
}

int cv_yolov8n_ob_deinit()
{
    return 0;
}

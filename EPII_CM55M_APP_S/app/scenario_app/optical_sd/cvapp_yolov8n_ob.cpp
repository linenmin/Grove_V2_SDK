/*
 * cvapp.cpp
 *
 *  Created on: 2018�~12��4��
 *      Author: 902452
 */

#include <cstdio>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "WE2_device.h"
#include "board.h"
#include "cvapp_yolov8n_ob.h"
#include "cisdp_sensor.h"

#include "WE2_core.h"

#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif
#include "img_proc_helium.h"
#include "yolo_postprocessing.h"


#include "xprintf.h"
#include "spi_master_protocol.h"
#include "cisdp_cfg.h"
#include "memory_manage.h"
#include <send_result.h>

#define INPUT_IMAGE_CHANNELS 6       // 光流模型输入通道：两帧 RGB 拼成 6 通道
#define YOLOV8_OB_INPUT_TENSOR_WIDTH   256   // 与模型输入一致：宽 256
#define YOLOV8_OB_INPUT_TENSOR_HEIGHT  192   // 与模型输入一致：高 192
#define YOLOV8_OB_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS

#define YOLOV8N_OB_DBG_APP_LOG 0     // 简单关闭调试日志


// #define EACH_STEP_TICK
#define TOTAL_STEP_TICK
#define YOLOV8_POST_EACH_STEP_TICK 0
uint32_t systick_1, systick_2;
uint32_t loop_cnt_1, loop_cnt_2;
#define CPU_CLK	0xffffff+1
static uint32_t capture_image_tick = 0;
#ifdef TRUSTZONE_SEC
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE	BASE_ADDR_APB_U55_CTRL
#endif
#endif


using namespace std;

namespace {

constexpr int tensor_arena_size = 1053*1024;

static uint32_t tensor_arena=0;

struct ethosu_driver ethosu_drv; /* Default Ethos-U device driver */
tflite::MicroInterpreter *yolov8n_ob_int_ptr=nullptr;
TfLiteTensor *yolov8n_ob_input, *yolov8n_ob_output;
};

static void _arm_npu_irq_handler(void)
{
    /* Call the default interrupt handler from the NPU driver */
    ethosu_irq_handler(&ethosu_drv);
}

/**
 * @brief  Initialises the NPU IRQ
 **/
static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;

    /* Register the EthosU IRQ handler in our vector table.
     * Note, this handler comes from the EthosU driver */
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);

    /* Enable the IRQ */
    NVIC_EnableIRQ(ethosu_irqnum);

}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;

    /* Initialise the IRQ */
    _arm_npu_irq_init();

    /* Initialise Ethos-U55 device */
#if TFLM2209_U55TAG2205
	const void * ethosu_base_address = (void *)(U55_BASE);
#else 
	void * const ethosu_base_address = (void *)(U55_BASE);
#endif

    if (0 != (err = ethosu_init(
                            &ethosu_drv,             /* Ethos-U driver device pointer */
                            ethosu_base_address,     /* Ethos-U NPU's base address. */
                            NULL,       /* Pointer to fast mem area - NULL for U55. */
                            0, /* Fast mem region size. */
							security_enable,                       /* Security enable. */
							privilege_enable))) {                   /* Privilege enable. */
    	xprintf("failed to initalise Ethos-U device\n");
            return err;
        }

    xprintf("Ethos-U55 device initialised\n");

    return 0;
}


int cv_yolov8n_ob_init(bool security_enable, bool privilege_enable, uint32_t model_addr) {
	int ercode = 0;

	// 申请张量内存
	tensor_arena = mm_reserve_align(tensor_arena_size,0x20); //1MB
	xprintf("TA[%x]\r\n",tensor_arena);

	// 初始化 Ethos-U
	if(_arm_npu_init(security_enable, privilege_enable)!=0)
		return -1;

	if(model_addr != 0) {
		static const tflite::Model*yolov8n_ob_model = tflite::GetModel((const void *)model_addr);

		if (yolov8n_ob_model->version() != TFLITE_SCHEMA_VERSION) {
			xprintf("[ERROR] model schema %d != %d\n", yolov8n_ob_model->version(), TFLITE_SCHEMA_VERSION);
			return -1;
		} else {
			xprintf("model schema %d\n", yolov8n_ob_model->version());
		}
#if TFLM2209_U55TAG2205
		static tflite::MicroErrorReporter yolov8n_ob_micro_error_reporter;
#endif
		static tflite::MicroMutableOpResolver<2> yolov8n_ob_op_resolver;

		yolov8n_ob_op_resolver.AddTranspose();
		if (kTfLiteOk != yolov8n_ob_op_resolver.AddEthosU()){
			xprintf("Failed to add Arm NPU support to op resolver.");
			return false;
		}
#if TFLM2209_U55TAG2205
		static tflite::MicroInterpreter yolov8n_ob_static_interpreter(yolov8n_ob_model, yolov8n_ob_op_resolver,
				(uint8_t*)tensor_arena, tensor_arena_size, &yolov8n_ob_micro_error_reporter);
#else
		static tflite::MicroInterpreter yolov8n_ob_static_interpreter(yolov8n_ob_model, yolov8n_ob_op_resolver,
				(uint8_t*)tensor_arena, tensor_arena_size);  
#endif  

		if(yolov8n_ob_static_interpreter.AllocateTensors()!= kTfLiteOk) {
			return false;
		}
		yolov8n_ob_int_ptr = &yolov8n_ob_static_interpreter;
		yolov8n_ob_input = yolov8n_ob_static_interpreter.input(0);
		yolov8n_ob_output = yolov8n_ob_static_interpreter.output(0);
	}

	xprintf("initial done\n");
	return ercode;
}



// --------------- 光流：无 NMS，直接取输出张量 ----------------

int cv_yolov8n_ob_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob) {
	int ercode = 0;
    uint32_t img_w = app_get_raw_width();
    uint32_t img_h = app_get_raw_height();
    uint32_t ch = app_get_raw_channels();
    uint32_t raw_addr = app_get_raw_addr();

	// 结果结构清零，防止旧数据
	memset(algoresult_yolov8n_ob, 0, sizeof(struct_yolov8_ob_algoResult));

    if(yolov8n_ob_int_ptr!= nullptr) {
		// resize 到模型输入分辨率（先得到 3 通道 RGB）
		static uint8_t resized_rgb[YOLOV8_OB_INPUT_TENSOR_WIDTH*YOLOV8_OB_INPUT_TENSOR_HEIGHT*3];
		float w_scale = (float)(img_w - 1) / (YOLOV8_OB_INPUT_TENSOR_WIDTH - 1);
		float h_scale = (float)(img_h - 1) / (YOLOV8_OB_INPUT_TENSOR_HEIGHT - 1);
		hx_lib_image_resize_BGR8U3C_to_RGB24_helium((uint8_t*)raw_addr, resized_rgb,  
		                    img_w, img_h, ch, 
                        	YOLOV8_OB_INPUT_TENSOR_WIDTH, YOLOV8_OB_INPUT_TENSOR_HEIGHT, w_scale,h_scale);

		// 构造 6 通道输入：简单用同一帧复制两次（RGB|RGB）
		int8_t *input_ptr = (int8_t *)yolov8n_ob_input->data.data;
		size_t pix_cnt = YOLOV8_OB_INPUT_TENSOR_WIDTH*YOLOV8_OB_INPUT_TENSOR_HEIGHT;
		for (size_t i = 0; i < pix_cnt*3; ++i) {
			int16_t q = (int16_t)resized_rgb[i] - 128; // zero_point = -128, scale = 1/255
			if (q > 127) q = 127;
			if (q < -128) q = -128;
			// 第一帧 RGB
			input_ptr[i] = (int8_t)q;
			// 第二帧 RGB 紧接着放在后 3 通道
			input_ptr[i + pix_cnt*3] = (int8_t)q;
    	}

		// 推理
		TfLiteStatus invoke_status = yolov8n_ob_int_ptr->Invoke();
		if(invoke_status != kTfLiteOk)
		{
			xprintf("optical flow invoke fail\n");
			return -1;
		}

		// 读取输出：shape [1, H, W, 4]，通道 0/1 为 dx/dy，2/3 为 log_var
		float out_scale = ((TfLiteAffineQuantization*)(yolov8n_ob_output->quantization.params))->scale->data[0];
		int out_zp = ((TfLiteAffineQuantization*)(yolov8n_ob_output->quantization.params))->zero_point->data[0];
		int8_t *out_data = yolov8n_ob_output->data.int8;

		// 示例：打印中心点光流值（可自行改为写 SD）
		size_t center_idx = (YOLOV8_OB_INPUT_TENSOR_HEIGHT/2)*YOLOV8_OB_INPUT_TENSOR_WIDTH + (YOLOV8_OB_INPUT_TENSOR_WIDTH/2);
		float dx = ((float)out_data[center_idx*4 + 0] - out_zp) * out_scale;
		float dy = ((float)out_data[center_idx*4 + 1] - out_zp) * out_scale;
		xprintf("flow center dx=%f dy=%f\r\n", dx, dy);

		// 如需写入 SD，请在此处将 out_data 直接写文件，或反量化后写 float。
    }

	// 重新触发下一帧
	sensordplib_retrigger_capture();
	return ercode;
}

int cv_yolov8n_ob_deinit()
{
	
	return 0;
}


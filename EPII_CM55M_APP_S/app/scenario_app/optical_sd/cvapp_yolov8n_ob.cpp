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
#include "ff.h" // FatFS 读写 raw
#include "hx_drv_gpio.h" // SPI CS 引脚控制
#include "hx_drv_scu.h"

#define INPUT_IMAGE_CHANNELS 6       // 光流模型输入通道：两帧 RGB 拼成 6 通道
#define YOLOV8_OB_INPUT_TENSOR_WIDTH   256   // 与模型输入一致：宽 256
#define YOLOV8_OB_INPUT_TENSOR_HEIGHT  192   // 与模型输入一致：高 192
#define YOLOV8_OB_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS

#define CPU_CLK	0xffffff+1  // SysTick 24-bit 回退路径使用
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

// 提供 FatFS SPI 端口需要的 GPIO 控制函数（与 fatfs_test.c 保持一致）
extern "C" {
void SSPI_CS_GPIO_Output_Level(bool setLevelHigh)
{
    hx_drv_gpio_set_out_value(GPIO16, (GPIO_OUT_LEVEL_E)setLevelHigh);
}

void SSPI_CS_GPIO_Pinmux(bool setGpioFn)
{
    if (setGpioFn)
        hx_drv_scu_set_PB5_pinmux(SCU_PB5_PINMUX_GPIO16, 0);
    else
        hx_drv_scu_set_PB5_pinmux(SCU_PB5_PINMUX_SPI_M_CS_1, 0);
}

void SSPI_CS_GPIO_Dir(bool setDirOut)
{
    if (setDirOut)
        hx_drv_gpio_set_output(GPIO16, GPIO_OUT_HIGH);
    else
        hx_drv_gpio_set_input(GPIO16);
}
}

namespace {
//vela报告中的内存峰值：1169.625 KB
constexpr int tensor_arena_size = 1600*1024;

static uint32_t tensor_arena=0;

struct ethosu_driver ethosu_drv; /* Default Ethos-U device driver */
tflite::MicroInterpreter *yolov8n_ob_int_ptr=nullptr;
TfLiteTensor *yolov8n_ob_input, *yolov8n_ob_output;
static uint8_t *raw_buf_1 = nullptr; // SD 帧缓存1（从内存池分配，避免占用BSS）
static uint8_t *raw_buf_2 = nullptr; // SD 帧缓存2
static uint32_t raw_buf_1_sum = 0, raw_buf_2_sum = 0; // 简单校验和
static uint8_t raw_buf_1_min = 0, raw_buf_1_max = 0; // 最值
static uint8_t raw_buf_2_min = 0, raw_buf_2_max = 0;
};

// SD raw 读取相关配置
// 你的 raw 放在 PC 路径 e:\ai_master\dataset\sintel\test\clean\market_4\raw\frame_0001_rgb888.raw
// 拷贝到 SD 后，假设目录为 0:/market_4/raw/
static const char *RAW_DIR = "0:/ai_master/dataset/sintel/test/clean/market_4/raw";             // SD 卡目录
static const char *RAW_FMT = "frame_%04d_rgb888.raw";       // 文件名模板
static const int RAW_FRAME_BYTES = YOLOV8_OB_INPUT_TENSOR_WIDTH * YOLOV8_OB_INPUT_TENSOR_HEIGHT * 3; // 256*192*3
static int raw_frame_idx = 1;                               // 当前帧序号
static const int raw_frame_max = 50;                        // 数据集中帧数（可按实际修改）
static int loop_cnt = 0;                                    // 推理循环计数
static const int LOG_PRINT_INTERVAL = 5;                    // 日志打印间隔：每5帧打印一次
static bool use_dwt_counter = false;                        // 是否启用 DWT 周期计数器
static uint32_t cm55m_clk_hz = 0;                           // CM55M 实际主频(Hz)
static FATFS fs;
static bool sd_mounted = false;

// 简单校验和与最值，排查输入/输出是否全零
static void compute_checksum(const uint8_t *buf, size_t len, uint32_t *sum_out, uint8_t *min_out, uint8_t *max_out)
{
	uint32_t s = 0;
	uint8_t mn = 0xFF, mx = 0;
	for (size_t i = 0; i < len; ++i) {
		uint8_t v = buf[i];
		s += v;
		if (v < mn) mn = v;
		if (v > mx) mx = v;
	}
	*sum_out = s;
	*min_out = mn;
	*max_out = mx;
}

// 初始化高精度计时：优先使用 DWT 周期计数器，避免 RTOS 下 SysTick 计时偏差
static void init_perf_counter()
{
	uint32_t freq = 0;
	if (hx_drv_scu_get_freq(SCU_CLK_FREQ_TYPE_HSC_CM55M, &freq) == SCU_NO_ERROR && freq > 0U) {
		cm55m_clk_hz = freq;
	} else if (SystemCoreClock > 0U) {
		cm55m_clk_hz = SystemCoreClock;
	} else {
		cm55m_clk_hz = 24000000U;
	}

	// 某些芯片/配置可能关闭了 DWT CYCCNT，先做可用性探测
	if ((DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) != 0U) {
		use_dwt_counter = false;
		xprintf("DWT CYCCNT unavailable, fallback to legacy tick timing\r\n");
		return;
	}

	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
	DWT->CYCCNT = 0U;
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
	__DSB();
	__ISB();

	uint32_t c0 = DWT->CYCCNT;
	for (volatile int i = 0; i < 64; ++i) { __NOP(); }
	uint32_t c1 = DWT->CYCCNT;
	use_dwt_counter = (c1 != c0);

	if (!use_dwt_counter) {
		xprintf("DWT CYCCNT start failed, fallback to legacy tick timing\r\n");
	}
}

// 将 cycle 数转换为 us（微秒），四舍五入
static uint32_t cycles_to_us(uint32_t cycles)
{
	if (cm55m_clk_hz == 0U) return 0U;
	uint64_t us = ((uint64_t)cycles * 1000000ULL + (uint64_t)(cm55m_clk_hz / 2U)) / (uint64_t)cm55m_clk_hz;
	if (us > 0xFFFFFFFFULL) us = 0xFFFFFFFFULL;
	return (uint32_t)us;
}

// 统一计时打点结构：DWT 走 tick 字段，SysTick 回退时使用 tick+loop
typedef struct {
	uint32_t tick;
	uint32_t loop;
} perf_stamp_t;

static uint32_t systick_ticks_to_us(uint32_t ticks)
{
	return (uint32_t)(((uint64_t)ticks * 1000000ULL + (CPU_CLK / 2U)) / CPU_CLK);
}

static inline void perf_mark(perf_stamp_t *s)
{
	if (use_dwt_counter) {
		s->tick = DWT->CYCCNT;
		s->loop = 0;
	} else {
		SystemGetTick(&s->tick, &s->loop);
	}
}

static uint32_t perf_elapsed_us(const perf_stamp_t *start, const perf_stamp_t *end)
{
	if (use_dwt_counter) {
		return cycles_to_us(end->tick - start->tick);
	}
	uint32_t ticks = (end->loop - start->loop) * CPU_CLK + (start->tick - end->tick);
	return systick_ticks_to_us(ticks);
}

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

	// 初始化性能计数器（用于更精确统计推理耗时）
	init_perf_counter();

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

	// 分配两帧 raw 缓冲区（256*192*3*2 ≈ 288 KB），从 mm 内存池取，避免挤占 256 KB DATA
	uint32_t buf_base = mm_reserve_align(RAW_FRAME_BYTES * 2, 0x20);
	if (buf_base == 0) {
		xprintf("alloc raw buffer fail\n");
		return -1;
	}
	raw_buf_1 = (uint8_t *)buf_base;
	raw_buf_2 = (uint8_t *)(buf_base + RAW_FRAME_BYTES);

	xprintf("initial done\n");
	return ercode;
}



// --------------- 光流：从 SD 读取 raw，填充 6 通道输入 ----------------

// 读取一帧 raw(RGB888) 到 dst_buf，返回 0 成功
static int load_raw_frame(int idx, uint8_t *dst_buf, size_t dst_size)
{
	char filepath[128];
	FIL fp;
	UINT br = 0;

	char namebuf[64];
	snprintf(namebuf, sizeof(namebuf), RAW_FMT, idx);
	snprintf(filepath, sizeof(filepath), "%s/%s", RAW_DIR, namebuf);

	if (!sd_mounted) {
		if (f_mount(&fs, "0:", 1) != FR_OK) {
			xprintf("SD mount fail\n");
			return -1;
		}
		sd_mounted = true;
	}

	if (f_open(&fp, filepath, FA_READ) != FR_OK) {
		xprintf("open %s fail\n", filepath);
		return -1;
	}
	if (f_read(&fp, dst_buf, dst_size, &br) != FR_OK || br != dst_size) {
		xprintf("read %s fail, br=%u\n", filepath, br);
		f_close(&fp);
		return -1;
	}
	f_close(&fp);
	return 0;
}

int cv_yolov8n_ob_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob) {
	int ercode = 0;

	// 清零结果
	memset(algoresult_yolov8n_ob, 0, sizeof(struct_yolov8_ob_algoResult));

    if(yolov8n_ob_int_ptr!= nullptr) {
		// 各阶段计时点
		perf_stamp_t t_total_start, t_total_end;
		perf_stamp_t t_sd_start, t_sd_end;
		perf_stamp_t t_preproc_start, t_preproc_end;
		perf_stamp_t t_infer_start, t_infer_end;
		uint32_t sd_us = 0, preproc_us = 0, infer_us = 0, total_us = 0;

		// 开始总计时 / SD计时
		perf_mark(&t_total_start);
		perf_mark(&t_sd_start);

		// 读取两帧 raw（RGB888），前帧 idx，后帧 idx+1
		if (raw_buf_1 == nullptr || raw_buf_2 == nullptr) {
			xprintf("raw buffers not allocated\n");
			return -1;
		}
		if (load_raw_frame(raw_frame_idx, raw_buf_1, RAW_FRAME_BYTES) != 0) {
			return -1;
		}
		int next_idx = raw_frame_idx + 1;
		if (next_idx > raw_frame_max) next_idx = 1;
		if (load_raw_frame(next_idx, raw_buf_2, RAW_FRAME_BYTES) != 0) {
			return -1;
		}

		// 结束SD读取计时，开始预处理计时
		perf_mark(&t_sd_end);
		perf_mark(&t_preproc_start);

		// 计算输入帧的简单校验和与最值
		compute_checksum(raw_buf_1, RAW_FRAME_BYTES, &raw_buf_1_sum, &raw_buf_1_min, &raw_buf_1_max);
		compute_checksum(raw_buf_2, RAW_FRAME_BYTES, &raw_buf_2_sum, &raw_buf_2_min, &raw_buf_2_max);

		// 填充 6 通道输入并量化到 int8（zero_point=-128, scale=1/255）
		int8_t *input_ptr = (int8_t *)yolov8n_ob_input->data.data;
		size_t pix_cnt = YOLOV8_OB_INPUT_TENSOR_WIDTH*YOLOV8_OB_INPUT_TENSOR_HEIGHT;
		for (size_t i = 0; i < pix_cnt*3; ++i) {
			int16_t q1 = (int16_t)raw_buf_1[i] - 128;
			if (q1 > 127) q1 = 127;
			if (q1 < -128) q1 = -128;
			input_ptr[i] = (int8_t)q1;

			int16_t q2 = (int16_t)raw_buf_2[i] - 128;
			if (q2 > 127) q2 = 127;
			if (q2 < -128) q2 = -128;
			input_ptr[i + pix_cnt*3] = (int8_t)q2;
    	}

		// 结束预处理计时，开始推理计时
		perf_mark(&t_preproc_end);
		perf_mark(&t_infer_start);

		// 推理
		TfLiteStatus invoke_status = yolov8n_ob_int_ptr->Invoke();
		if(invoke_status != kTfLiteOk)
		{
			xprintf("optical flow invoke fail\n");
			return -1;
		}

		// 结束推理计时并计算各阶段耗时
		perf_mark(&t_infer_end);
		perf_mark(&t_total_end);

		sd_us = perf_elapsed_us(&t_sd_start, &t_sd_end);
		preproc_us = perf_elapsed_us(&t_preproc_start, &t_preproc_end);
		infer_us = perf_elapsed_us(&t_infer_start, &t_infer_end);
		total_us = perf_elapsed_us(&t_total_start, &t_total_end);

		// 读取输出：shape [1, H, W, 4]，通道 0/1 为 dx/dy，2/3 为 log_var
		float out_scale = ((TfLiteAffineQuantization*)(yolov8n_ob_output->quantization.params))->scale->data[0];
		int out_zp = ((TfLiteAffineQuantization*)(yolov8n_ob_output->quantization.params))->zero_point->data[0];
		int8_t *out_data = yolov8n_ob_output->data.int8;

		// 示例1：中心点光流（整数形式，避免 %f）
		size_t center_idx = (YOLOV8_OB_INPUT_TENSOR_HEIGHT/2)*YOLOV8_OB_INPUT_TENSOR_WIDTH + (YOLOV8_OB_INPUT_TENSOR_WIDTH/2);
		float dx_f = ((float)out_data[center_idx*4 + 0] - out_zp) * out_scale;
		float dy_f = ((float)out_data[center_idx*4 + 1] - out_zp) * out_scale;
		int dx_milli = (int)(dx_f * 1000); // 放大1000倍打印三位小数
		int dy_milli = (int)(dy_f * 1000);

		// 示例2：全局平均光流（先累加 int，再乘 scale）
		const int out_h = YOLOV8_OB_INPUT_TENSOR_HEIGHT;
		const int out_w = YOLOV8_OB_INPUT_TENSOR_WIDTH;
		int64_t sum_dx = 0;
		int64_t sum_dy = 0;
		const int pixels = out_h * out_w;
		for (int i = 0; i < pixels; ++i) {
			sum_dx += (int)out_data[i*4 + 0] - out_zp;
			sum_dy += (int)out_data[i*4 + 1] - out_zp;
		}
		float mean_dx = (pixels > 0) ? ((float)sum_dx / (float)pixels) * out_scale : 0.f;
		float mean_dy = (pixels > 0) ? ((float)sum_dy / (float)pixels) * out_scale : 0.f;
		int mean_dx_milli = (int)(mean_dx * 1000);
		int mean_dy_milli = (int)(mean_dy * 1000);

		// 按日志间隔打印，减少串口输出对循环的影响。
		if ((loop_cnt % LOG_PRINT_INTERVAL) == 0) {
			// 输出加入输入校验和与输出最值，排查全零
			int8_t out0_min = 127, out0_max = -128;
			int8_t out1_min = 127, out1_max = -128;
			for (int i = 0; i < pixels; ++i) {
				int8_t v0 = out_data[i*4 + 0];
				int8_t v1 = out_data[i*4 + 1];
				if (v0 < out0_min) out0_min = v0;
				if (v0 > out0_max) out0_max = v0;
				if (v1 < out1_min) out1_min = v1;
				if (v1 > out1_max) out1_max = v1;
			}

			xprintf("[loop=%d][frame=%d/%d] center dx=%d.%03d dy=%d.%03d | mean dx=%d.%03d dy=%d.%03d | in1 sum=%u min=%u max=%u | in2 sum=%u min=%u max=%u | out0 min=%d max=%d out1 min=%d max=%d | times: sd=%u.%03ums preproc=%u.%03ums infer=%u.%03ums total=%u.%03ums\r\n",
					loop_cnt,
					raw_frame_idx, raw_frame_max,
					dx_milli/1000, (dx_milli>=0?dx_milli:-dx_milli)%1000,
					dy_milli/1000, (dy_milli>=0?dy_milli:-dy_milli)%1000,
					mean_dx_milli/1000, (mean_dx_milli>=0?mean_dx_milli:-mean_dx_milli)%1000,
					mean_dy_milli/1000, (mean_dy_milli>=0?mean_dy_milli:-mean_dy_milli)%1000,
					raw_buf_1_sum, raw_buf_1_min, raw_buf_1_max,
					raw_buf_2_sum, raw_buf_2_min, raw_buf_2_max,
					out0_min, out0_max, out1_min, out1_max,
					sd_us/1000, sd_us%1000,
					preproc_us/1000, preproc_us%1000,
					infer_us/1000, infer_us%1000,
					total_us/1000, total_us%1000);
		}

		// 帧序号前进
		raw_frame_idx++;
		if (raw_frame_idx > raw_frame_max) raw_frame_idx = 1;
		loop_cnt++;
    }

	// 不使用摄像头，故不再触发采集
	return ercode;
}

int cv_yolov8n_ob_deinit()
{
	
	return 0;
}

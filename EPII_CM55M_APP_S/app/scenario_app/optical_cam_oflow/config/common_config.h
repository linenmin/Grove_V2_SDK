/*
 * common_config.h
 *
 *  Created on: Nov 22, 2022
 *      Author: bigcat-himax
 */


#ifndef SCENARIO_TFLM_2IN1_FD_FL_PL_COMMON_CONFIG_H_
#define SCENARIO_TFLM_2IN1_FD_FL_PL_COMMON_CONFIG_H_

#define FRAME_CHECK_DEBUG	1
#define EN_ALGO				1
// #define SPI_SEN_PIC_CLK				(10000000)
#define SPI_SEN_PIC_CLK				(12000000)


#define DBG_APP_LOG 0

//current FW image is 409600 bytes => 0x64000. set  0~0x171000 as FW area
#define FW_IMG_SZ							0x3A171000




//0x3AB7B000 //(2220032 bytes => 0x21E000, set to 0x21E000)
#define YOLOV8_OBJECT_DETECTION_FLASH_ADDR 0x3AB7B000

// --- Optical Flow Firmware Common Configuration ---

// Centralized Model Parameters
// Modify these when changing the Vela exported model
#define FLOW_MODEL_IN_W 192
#define FLOW_MODEL_IN_H 144
#define FLOW_MODEL_CHANNELS 6

// Tensor Arena Size
// D8/D16: Minimum 1432KB for 150x200 or 144x192 models.
#define FLOW_TENSOR_ARENA_KB 1432

// Visualization Mode
// 1 = Color HSV (Hue=direction, Value=magnitude)
// 0 = Grayscale (magnitude only)
#define FLOW_VIZ_RGB_OUTPUT 1


#endif /* SCENARIO_TFLM_2IN1_FD_FL_PL_COMMON_CONFIG_H_ */

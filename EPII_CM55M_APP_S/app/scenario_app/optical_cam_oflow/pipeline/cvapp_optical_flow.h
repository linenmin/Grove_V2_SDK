/*
 * cvapp.h
 *
 *  Created on: 2018�~12��4��
 *      Author: 902452
 */

#ifndef SCENARIO_TFLM_2IN1_FD_FL_PL_CVAPP_PL_
#define SCENARIO_TFLM_2IN1_FD_FL_PL_CVAPP_PL_

#include <stdbool.h>

#include "spi_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

int cv_optical_flow_init(bool security_enable, bool privilege_enable, uint32_t model_addr);

int cv_optical_flow_run(struct_optical_flow_algoResult *algoresult_optical_flow);

int cv_optical_flow_deinit();
#ifdef __cplusplus
}
#endif

#endif /* SCENARIO_TFLM_2IN1_FD_FL_PL_CVAPP_PL_ */

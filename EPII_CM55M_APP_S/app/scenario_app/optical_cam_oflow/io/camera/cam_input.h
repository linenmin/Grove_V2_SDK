#ifndef OPTICAL_CAM_OFLOW_CAM_INPUT_H_
#define OPTICAL_CAM_OFLOW_CAM_INPUT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int cam_input_init(uint32_t model_w, uint32_t model_h);
int cam_input_get_frame(uint8_t *frame, size_t bytes_per_frame);
void cam_input_deinit(void);

#ifdef __cplusplus
}
#endif

#endif  // OPTICAL_CAM_OFLOW_CAM_INPUT_H_

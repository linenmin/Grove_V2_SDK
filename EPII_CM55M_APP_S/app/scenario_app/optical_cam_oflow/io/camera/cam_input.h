#ifndef OPTICAL_CAM_OFLOW_CAM_INPUT_H_
#define OPTICAL_CAM_OFLOW_CAM_INPUT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int cam_input_init(void);
int cam_input_get_frame_pair(uint8_t *frame_t, uint8_t *frame_t1, size_t bytes_per_frame);
void cam_input_deinit(void);

#ifdef __cplusplus
}
#endif

#endif  // OPTICAL_CAM_OFLOW_CAM_INPUT_H_

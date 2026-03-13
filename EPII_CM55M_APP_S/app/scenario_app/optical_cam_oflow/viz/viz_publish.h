#ifndef VIZ_VIZ_PUBLISH_H
#define VIZ_VIZ_PUBLISH_H

#include <stdint.h>
#include <stddef.h>
#include "cvapp_optical_flow.h"

#ifdef __cplusplus
extern "C" {
#endif

int viz_publish_init(int flow_w, int flow_h);

void publish_viz_payload(struct_optical_flow_algoResult *algo,
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
                        int input_h);

#ifdef __cplusplus
}
#endif

#endif // VIZ_VIZ_PUBLISH_H

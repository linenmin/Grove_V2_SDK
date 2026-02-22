#ifndef OPTICAL_SD_OB_PERF_H_
#define OPTICAL_SD_OB_PERF_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t tick;
    uint32_t loop;
} ob_perf_stamp_t;

void ob_perf_init(void);
void ob_perf_mark(ob_perf_stamp_t *stamp);
uint32_t ob_perf_elapsed_us(const ob_perf_stamp_t *start, const ob_perf_stamp_t *end);

#ifdef __cplusplus
}
#endif

#endif  // OPTICAL_SD_OB_PERF_H_

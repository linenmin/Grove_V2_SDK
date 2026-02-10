#ifndef OPTICAL_SD_OB_SD_FRAME_H_
#define OPTICAL_SD_OB_SD_FRAME_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int ob_sd_init(const char *raw_dir, const char *raw_fmt, int frame_max);
int ob_sd_load_frame(int frame_idx, uint8_t *buf, size_t bytes_per_frame);
int ob_sd_next_frame_idx(int frame_idx, int *next_idx);

#ifdef __cplusplus
}
#endif

#endif  // OPTICAL_SD_OB_SD_FRAME_H_

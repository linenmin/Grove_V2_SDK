#include "ob_sd_frame.h"

#include <stdio.h>

#include "ff.h"
#include "xprintf.h"

namespace {

static const char *g_raw_dir = nullptr;
static const char *g_raw_fmt = nullptr;
static int g_frame_max = 0;

static FATFS g_fs;
static bool g_sd_mounted = false;

}  // namespace

int ob_sd_init(const char *raw_dir, const char *raw_fmt, int frame_max)
{
    if (raw_dir == nullptr || raw_fmt == nullptr || frame_max <= 0) {
        return -1;
    }

    g_raw_dir = raw_dir;
    g_raw_fmt = raw_fmt;
    g_frame_max = frame_max;
    g_sd_mounted = false;
    return 0;
}

int ob_sd_next_frame_idx(int frame_idx, int *next_idx)
{
    if (next_idx == nullptr || g_frame_max <= 0) {
        return -1;
    }

    int idx = frame_idx + 1;
    if (idx > g_frame_max) {
        idx = 1;
    }
    *next_idx = idx;
    return 0;
}

int ob_sd_load_frame(int frame_idx, uint8_t *buf, size_t bytes_per_frame)
{
    if (g_raw_dir == nullptr || g_raw_fmt == nullptr || g_frame_max <= 0) {
        xprintf("SD config invalid\n");
        return -1;
    }
    if (buf == nullptr || bytes_per_frame == 0) {
        return -1;
    }

    char filepath[128];
    char namebuf[64];
    FIL fp;
    UINT br = 0;

    snprintf(namebuf, sizeof(namebuf), g_raw_fmt, frame_idx);
    snprintf(filepath, sizeof(filepath), "%s/%s", g_raw_dir, namebuf);

    if (!g_sd_mounted) {
        if (f_mount(&g_fs, "0:", 1) != FR_OK) {
            xprintf("SD mount fail\n");
            return -1;
        }
        g_sd_mounted = true;
    }

    if (f_open(&fp, filepath, FA_READ) != FR_OK) {
        xprintf("open %s fail\n", filepath);
        return -1;
    }

    if (f_read(&fp, buf, bytes_per_frame, &br) != FR_OK || br != bytes_per_frame) {
        xprintf("read %s fail, br=%u\n", filepath, br);
        f_close(&fp);
        return -1;
    }

    f_close(&fp);
    return 0;
}

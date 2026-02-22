#include <stdint.h>
#include <stdbool.h>

#include "WE2_core.h"
#include "WE2_device.h"
#include "board.h"
#include "pinmux_init.h"
#include "common_config.h"
#include "cvapp_yolov8n_ob.h"
#include "hx_drv_pmu.h"
#include "hx_drv_scu.h"
#include "hx_drv_swreg_aon.h"
#include "memory_manage.h"
#include "powermode.h"
#include "spi_eeprom_comm.h"
#include "tflm_yolov8_od.h"
#include "xprintf.h"

static void optical_sd_runtime_init(void)
{
    uint32_t wakeup_event = 0;
    uint32_t wakeup_event1 = 0;
    uint32_t freq = 0;

    hx_drv_pmu_get_ctrl(PMU_pmu_wakeup_EVT, &wakeup_event);
    hx_drv_pmu_get_ctrl(PMU_pmu_wakeup_EVT1, &wakeup_event1);
    hx_drv_swreg_aon_get_pllfreq(&freq);
    xprintf("wakeup_event=0x%x,WakeupEvt1=0x%x, freq=%d\n", wakeup_event, wakeup_event1, freq);

    pinmux_init();

    if (!((wakeup_event == PMU_WAKEUP_NONE) && (wakeup_event1 == PMU_WAKEUPEVENT1_NONE))) {
        hx_lib_pm_ctrl_fromPMUtoCPU(NULL);
    }

    hx_lib_spi_eeprom_open(USE_DW_SPI_MST_Q);
    hx_lib_spi_eeprom_enable_XIP(USE_DW_SPI_MST_Q, true, FLASH_QUAD, true);

    xprintf("ori_clk src info, 0x56100030=%x\n", EPII_get_memory(0x56100030));
    xprintf("ori_clk src info, 0x56100034=%x\n", EPII_get_memory(0x56100034));
    xprintf("ori_clk src info, 0x56100038=%x\n", EPII_get_memory(0x56100038));

    EPII_set_memory(0x56100030, 0x4037);
    EPII_set_memory(0x56100034, 0x0);
    EPII_set_memory(0x56100038, 0xc1b8);

    xprintf("clk src info, 0x56100030=%x\n", EPII_get_memory(0x56100030));
    xprintf("clk src info, 0x56100034=%x\n", EPII_get_memory(0x56100034));
    xprintf("clk src info, 0x56100038=%x\n", EPII_get_memory(0x56100038));

#ifdef __GNU__
    extern char __mm_start_addr__;
    xprintf("__GNUC\n");
    xprintf("__mm_start_addr__ address: %x\r\n", &__mm_start_addr__);
    mm_set_initial((int)(&__mm_start_addr__),
                   0x00200000 - ((int)(&__mm_start_addr__) - 0x34000000));
#else
    static uint8_t mm_start_addr __attribute__((section(".bss.mm_start_addr")));
    xprintf("mm_start_addr address: %x\r\n", &mm_start_addr);
    mm_set_initial((int)(&mm_start_addr),
                   0x00200000 - ((int)(&mm_start_addr) - 0x34000000));
#endif
}

int tflm_yolov8_od_app(void)
{
    struct_yolov8_ob_algoResult result;

    optical_sd_runtime_init();
    xprintf("Optical SD clean app start\n");

    if (cv_yolov8n_ob_init(true, true, YOLOV8_OBJECT_DETECTION_FLASH_ADDR) != 0) {
        xprintf("cv_yolov8n_ob_init fail\n");
        APP_BLOCK_FUNC();
    }

    while (1) {
        if (cv_yolov8n_ob_run(&result) != 0) {
            xprintf("cv_yolov8n_ob_run fail\n");
        }
    }
}

void model_change(void)
{
    // SD-only clean app 不支持运行时模型切换，保留空实现以兼容接口。
}

void SetPSPDNoVid_24M(void) {}
void SetPSPDNoVid(void) {}

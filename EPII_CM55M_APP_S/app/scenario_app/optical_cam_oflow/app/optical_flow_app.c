#include <stdint.h>
#include <stdbool.h>

#include "WE2_core.h"
#include "WE2_device.h"
#include "board.h"
#include "pinmux_init.h"
#include "common_config.h"
#include "cvapp_optical_flow.h"
#include "hx_drv_pmu.h"
#include "hx_drv_spi.h"
#include "hx_drv_scu.h"
#include "hx_drv_swreg_aon.h"
#include "memory_manage.h"
#include "powermode.h"
#include "spi_master_protocol.h"
#include "spi_eeprom_comm.h"
#include "optical_flow_app.h"
#include "xprintf.h"

static void spi_m_pinmux_cfg_for_viz(void)
{
    SCU_PINMUX_CFG_T pinmux_cfg;
    hx_drv_scu_get_all_pinmux_cfg(&pinmux_cfg);
    pinmux_cfg.pin_pb2 = SCU_PB2_PINMUX_SPI_M_DO_1;
    pinmux_cfg.pin_pb3 = SCU_PB3_PINMUX_SPI_M_DI_1;
    pinmux_cfg.pin_pb4 = SCU_PB4_PINMUX_SPI_M_SCLK_1;
    pinmux_cfg.pin_pb11 = SCU_PB11_PINMUX_SPI_M_CS;
    hx_drv_scu_set_all_pinmux_cfg(&pinmux_cfg, 1);
}

static void optical_cam_runtime_init(void)
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

    spi_m_pinmux_cfg_for_viz();
    if (hx_drv_spi_mst_open_speed(SPI_SEN_PIC_CLK) != 0) {
        xprintf("viz spi master init fail\n");
    } else {
        xprintf("viz spi master init ok, clk=%d\n", SPI_SEN_PIC_CLK);
    }

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

int optical_flow_app(void)
{
    struct_optical_flow_algoResult result;

    optical_cam_runtime_init();
    xprintf("Optical camera oflow app start\n");

    if (cv_optical_flow_init(true, true, OPTICAL_FLOW_MODEL_FLASH_ADDR) != 0) {
        xprintf("cv_optical_flow_init fail\n");
        APP_BLOCK_FUNC();
    }

    while (1) {
        if (cv_optical_flow_run(&result) != 0) {
            xprintf("cv_optical_flow_run fail\n");
        }
    }
}

void model_change(void)
{
    // Camera M1 不支持运行时模型切换，保留空实现以兼容接口。
}

void SetPSPDNoVid_24M(void) {}
void SetPSPDNoVid(void) {}

#include "ob_perf.h"

#include "WE2_device.h"
#include "board.h"
#include "hx_drv_scu.h"
#include "xprintf.h"

#define CPU_CLK (0xFFFFFFU + 1U)

namespace {

static bool g_use_dwt_counter = false;
static uint32_t g_cm55m_clk_hz = 0;

static uint32_t cycles_to_us(uint32_t cycles)
{
    if (g_cm55m_clk_hz == 0U) {
        return 0U;
    }

    uint64_t us = ((uint64_t)cycles * 1000000ULL + (uint64_t)(g_cm55m_clk_hz / 2U)) /
                  (uint64_t)g_cm55m_clk_hz;
    if (us > 0xFFFFFFFFULL) {
        us = 0xFFFFFFFFULL;
    }
    return (uint32_t)us;
}

static uint32_t systick_ticks_to_us(uint32_t ticks)
{
    return (uint32_t)(((uint64_t)ticks * 1000000ULL + (CPU_CLK / 2U)) / CPU_CLK);
}

}  // namespace

void ob_perf_init(void)
{
    uint32_t freq = 0;
    if (hx_drv_scu_get_freq(SCU_CLK_FREQ_TYPE_HSC_CM55M, &freq) == SCU_NO_ERROR && freq > 0U) {
        g_cm55m_clk_hz = freq;
    } else if (SystemCoreClock > 0U) {
        g_cm55m_clk_hz = SystemCoreClock;
    } else {
        g_cm55m_clk_hz = 24000000U;
    }

    // 优先启用 DWT，失败后自动回退到旧计时方式。
    if ((DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) != 0U) {
        g_use_dwt_counter = false;
        xprintf("DWT CYCCNT unavailable, fallback to legacy tick timing\r\n");
        return;
    }

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __DSB();
    __ISB();

    const uint32_t c0 = DWT->CYCCNT;
    for (volatile int i = 0; i < 64; ++i) {
        __NOP();
    }
    const uint32_t c1 = DWT->CYCCNT;
    g_use_dwt_counter = (c1 != c0);

    if (!g_use_dwt_counter) {
        xprintf("DWT CYCCNT start failed, fallback to legacy tick timing\r\n");
    }
}

void ob_perf_mark(ob_perf_stamp_t *stamp)
{
    if (stamp == nullptr) {
        return;
    }

    if (g_use_dwt_counter) {
        stamp->tick = DWT->CYCCNT;
        stamp->loop = 0;
    } else {
        SystemGetTick(&stamp->tick, &stamp->loop);
    }
}

uint32_t ob_perf_elapsed_us(const ob_perf_stamp_t *start, const ob_perf_stamp_t *end)
{
    if (start == nullptr || end == nullptr) {
        return 0U;
    }

    if (g_use_dwt_counter) {
        return cycles_to_us(end->tick - start->tick);
    }

    const uint32_t ticks = (end->loop - start->loop) * CPU_CLK + (start->tick - end->tick);
    return systick_ticks_to_us(ticks);
}

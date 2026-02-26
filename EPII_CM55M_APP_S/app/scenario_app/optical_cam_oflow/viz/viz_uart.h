#ifndef OPTICAL_CAM_OFLOW_VIZ_UART_H_
#define OPTICAL_CAM_OFLOW_VIZ_UART_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** 轮询主机是否发送了 0xFF/0xFE/0xFD/0xFC（选择 UART/SPI/RAW 模式）；若收到则下次会立即发完整握手。 */
void viz_uart_poll_host_cmd(void);
/** 当前主机选择的传输模式：0=UART, 1=SPI, 2=UART+SPI, 3=RAW_UART。 */
uint8_t viz_uart_get_transport_mode(void);
void viz_uart_send_device_id_once(void);
void viz_uart_send_invoke_jpeg(const uint8_t *jpeg_data,
                               size_t jpeg_size,
                               uint16_t width,
                               uint16_t height,
                               uint32_t algo_tick_us);
/** RAW 模式：发送 8 字节二进制头 + 原始 JPEG 数据，无 Base64/JSON。 */
void viz_uart_send_raw_jpeg(const uint8_t *jpeg_data,
                           size_t jpeg_size,
                           uint16_t width,
                           uint16_t height);

#ifdef __cplusplus
}
#endif

#endif  // OPTICAL_CAM_OFLOW_VIZ_UART_H_

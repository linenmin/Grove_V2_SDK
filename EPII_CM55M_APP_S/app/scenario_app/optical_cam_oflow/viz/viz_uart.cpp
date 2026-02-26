#include "viz_uart.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include <string>

extern "C" {
#include "hx_drv_uart.h"
}
#include "xprintf.h"
#include "console_io.h"

constexpr int kConsoleUartId = 0;
constexpr size_t kHostCmdBufSize = 128U;
constexpr char kBase64Table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
constexpr char kNameData[] = "\"Grove Vision AI (WE2)\"";
constexpr char kVersionData[] =
    "{\"software\": \"optical_cam_oflow\", \"hardware\": \"WE2\"}";
constexpr char kInfoData[] = "{\"crc16_maxim\": 0, \"info\": \"\"}";

// 0=UART, 1=SPI, 2=UART+SPI。默认 UART，保证网页“uart send”即插即用。
static uint8_t g_transport_mode = 0U;
static bool g_invoke_enabled = true;
static uint8_t g_model_id = 0U;
static char g_host_cmd_buf[kHostCmdBufSize] = {0};
static size_t g_host_cmd_len = 0U;

// Custom putchar wrapper to suppress xprintf output in RAW mode
static void viz_uart_putchar(unsigned char c)
{
    if (g_transport_mode == 3U) {
        // In RAW binary mode (3), drop all text output to keep UART channel clean
        return;
    }
    // Otherwise, pass through to the default console output
    console_putchar(c);
}

static DEV_UART *uart_dev()
{
    DEV_UART *dev = hx_drv_uart_get_dev((USE_DW_UART_E)kConsoleUartId);
    if (dev != nullptr) {
        dev->uart_open(UART_BAUDRATE_921600);
    }
    return dev;
}

static void uart_send_bytes(const char *data, size_t len)
{
    DEV_UART *dev = uart_dev();
    if (dev == nullptr || data == nullptr || len == 0U) {
        return;
    }
    size_t sent = 0U;
    while (sent < len) {
        const size_t chunk = ((len - sent) < 16U) ? (len - sent) : 16U;
        sent += dev->uart_write(data + sent, chunk);
    }
}

static void uart_send(const std::string &s)
{
    if (s.empty()) {
        return;
    }
    uart_send_bytes(s.c_str(), s.size());
}

static void uart_send_literal(const char *s)
{
    if (s == nullptr || s[0] == '\0') {
        return;
    }
    uart_send_bytes(s, strlen(s));
}

static void uart_send_base64_stream(const uint8_t *data, size_t len)
{
    if (data == nullptr || len == 0U) {
        return;
    }

    char out[4U * 64U];
    size_t out_len = 0U;
    size_t i = 0U;
    while (i + 2U < len) {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16U) |
                           (static_cast<uint32_t>(data[i + 1U]) << 8U) |
                           static_cast<uint32_t>(data[i + 2U]);
        out[out_len++] = kBase64Table[(v >> 18U) & 0x3FU];
        out[out_len++] = kBase64Table[(v >> 12U) & 0x3FU];
        out[out_len++] = kBase64Table[(v >> 6U) & 0x3FU];
        out[out_len++] = kBase64Table[v & 0x3FU];
        i += 3U;

        if (out_len >= (sizeof(out) - 4U)) {
            uart_send_bytes(out, out_len);
            out_len = 0U;
        }
    }

    if (i < len) {
        uint32_t v = static_cast<uint32_t>(data[i]) << 16U;
        out[out_len++] = kBase64Table[(v >> 18U) & 0x3FU];
        if ((i + 1U) < len) {
            v |= static_cast<uint32_t>(data[i + 1U]) << 8U;
            out[out_len++] = kBase64Table[(v >> 12U) & 0x3FU];
            out[out_len++] = kBase64Table[(v >> 6U) & 0x3FU];
            out[out_len++] = '=';
        } else {
            out[out_len++] = kBase64Table[(v >> 12U) & 0x3FU];
            out[out_len++] = '=';
            out[out_len++] = '=';
        }
    }
    if (out_len > 0U) {
        uart_send_bytes(out, out_len);
    }
}

static std::string trim_copy(const std::string &in)
{
    size_t begin = 0U;
    while (begin < in.size() && (in[begin] == ' ' || in[begin] == '\t')) {
        begin++;
    }

    size_t end = in.size();
    while (end > begin && (in[end - 1U] == ' ' || in[end - 1U] == '\t')) {
        end--;
    }

    return in.substr(begin, end - begin);
}

static std::string upper_ascii_copy(const std::string &in)
{
    std::string out = in;
    for (size_t i = 0U; i < out.size(); ++i) {
        char c = out[i];
        if (c >= 'a' && c <= 'z') {
            out[i] = static_cast<char>(c - ('a' - 'A'));
        }
    }
    return out;
}

static uint32_t parse_first_uint(const std::string &args, uint32_t fallback)
{
    const size_t comma = args.find(',');
    const std::string first = trim_copy(args.substr(0U, comma));
    if (first.empty()) {
        return fallback;
    }

    char *endp = nullptr;
    const unsigned long v = strtoul(first.c_str(), &endp, 10);
    if (endp == first.c_str()) {
        return fallback;
    }
    return static_cast<uint32_t>(v);
}

static std::string model_data_json(void)
{
    std::string data;
    data.reserve(64U);
    data += "{\"id\": ";
    data += std::to_string(g_model_id);
    data += ", \"type\": 0, \"address\": 0, \"size\": 0}";
    return data;
}

static void send_type0(const std::string &name, const std::string &data, int code = 0)
{
    std::string msg;
    msg.reserve(name.size() + data.size() + 64U);
    msg += "\r{\"type\": 0, \"name\": \"";
    msg += name;
    msg += "\", \"code\": ";
    msg += std::to_string(code);
    msg += ", \"data\": ";
    msg += data;
    msg += "}\n";
    uart_send(msg);
}

static void send_full_device_id_sequence(void)
{
    send_type0("NAME?", kNameData);
    send_type0("VER?", kVersionData);
    send_type0("ID?", "1");
    send_type0("INFO?", kInfoData);
    send_type0("MODEL?", model_data_json());
}

static void handle_at_command(const std::string &line)
{
    const std::string raw = trim_copy(line);
    if (raw.size() < 4U) {
        return;
    }
    if (!(raw[0] == 'A' && raw[1] == 'T' && raw[2] == '+')) {
        return;
    }

    const std::string body = upper_ascii_copy(raw.substr(3U));
    if (body == "NAME?") {
        send_type0("NAME?", kNameData);
        return;
    }
    if (body == "VER?") {
        send_type0("VER?", kVersionData);
        return;
    }
    if (body == "ID?") {
        send_type0("ID?", "1");
        return;
    }
    if (body == "INFO?") {
        send_type0("INFO?", kInfoData);
        return;
    }
    if (body == "MODEL?") {
        send_type0("MODEL?", model_data_json());
        return;
    }
    if (body == "INVOKE?") {
        send_type0("INVOKE?", g_invoke_enabled ? "1" : "0");
        return;
    }
    if (body == "SAMPLE?") {
        send_type0("SAMPLE?", "0");
        return;
    }
    if (body == "STAT?") {
        send_type0("STAT?", "\"\"");
        return;
    }
    if (body.rfind("INVOKE=", 0U) == 0U) {
        const uint32_t enable = parse_first_uint(body.substr(7U), g_invoke_enabled ? 1U : 0U);
        g_invoke_enabled = (enable != 0U);
        send_type0("INVOKE", g_invoke_enabled ? "1" : "0");
        return;
    }
    if (body.rfind("MODEL=", 0U) == 0U) {
        g_model_id = static_cast<uint8_t>(parse_first_uint(body.substr(6U), g_model_id));
        send_type0("MODEL", model_data_json());
        return;
    }
    if (body.rfind("SAMPLE=", 0U) == 0U) {
        const uint32_t sample_state = parse_first_uint(body.substr(7U), 0U);
        send_type0("SAMPLE", sample_state ? "1" : "0");
        return;
    }
    if (body.rfind("INFO=", 0U) == 0U) {
        send_type0("INFO", "1");
        return;
    }
    if (body.rfind("LED=", 0U) == 0U) {
        send_type0("LED", "1");
        return;
    }

    const size_t eq_pos = body.find('=');
    if (eq_pos != std::string::npos) {
        send_type0(body.substr(0U, eq_pos), "0");
        return;
    }
    if (!body.empty() && body.back() == '?') {
        send_type0(body, "null");
        return;
    }
    send_type0(body, "0");
}

static void handle_host_byte(uint8_t byte)
{
    if (byte == 0xFFU) {
        g_transport_mode = 0U;
        xdev_out(console_putchar); // Restore text output
        send_full_device_id_sequence();
        g_host_cmd_len = 0U;
        return;
    }
    if (byte == 0xFEU) {
        g_transport_mode = 1U;
        xdev_out(console_putchar); // Restore text output
        send_full_device_id_sequence();
        g_host_cmd_len = 0U;
        return;
    }
    if (byte == 0xFDU) {
        g_transport_mode = 2U;
        xdev_out(console_putchar); // Restore text output
        send_full_device_id_sequence();
        g_host_cmd_len = 0U;
        return;
    }
    if (byte == 0xFCU) {
        g_transport_mode = 3U;  // RAW binary mode
        // Send a short ACK before suppressing text
        uart_send_literal("\r{\"type\": 0, \"name\": \"RAW_MODE\", \"code\": 0, \"data\": 1}\n");
        // Suppress all subsequent xprintf output to guarantee a clean binary channel
        xdev_out(viz_uart_putchar);
        g_host_cmd_len = 0U;
        return;
    }

    if (byte == '\r' || byte == '\n') {
        if (g_host_cmd_len > 0U) {
            const std::string line(g_host_cmd_buf, g_host_cmd_len);
            g_host_cmd_len = 0U;
            handle_at_command(line);
        }
        return;
    }

    if (!isprint(byte)) {
        return;
    }
    if ((g_host_cmd_len + 1U) >= kHostCmdBufSize) {
        g_host_cmd_len = 0U;
        return;
    }
    g_host_cmd_buf[g_host_cmd_len++] = static_cast<char>(byte);
}

// 主机点击 "uart send" 会发 0xFF；收到后立即回送完整握手，便于页面同步。
void viz_uart_poll_host_cmd(void)
{
    DEV_UART *dev = uart_dev();
    if (dev == nullptr) {
        return;
    }
    for (int i = 0; i < 64; ++i) {
        char c = 0;
        const int32_t n = dev->uart_read_nonblock(&c, 1);
        if (n != 1) {
            break;
        }
        handle_host_byte(static_cast<uint8_t>(c));
    }
}

uint8_t viz_uart_get_transport_mode(void)
{
    return g_transport_mode;
}

// 与官方 send_device_id 一致：NAME? VER? ID? INFO? MODEL? 五条握手。
void viz_uart_send_device_id_once(void)
{
    static uint32_t call_count = 0;
    call_count++;
    if (!(call_count == 1U || (call_count % 20U) == 0U)) {
        return;
    }
    send_full_device_id_sequence();
}

void viz_uart_send_invoke_jpeg(const uint8_t *jpeg_data,
                               size_t jpeg_size,
                               uint16_t width,
                               uint16_t height,
                               uint32_t algo_tick_us)
{
    if (!g_invoke_enabled || jpeg_data == nullptr || jpeg_size == 0U) {
        return;
    }

    std::string prefix;
    prefix.reserve(256U);
    prefix += "\r{\"type\": 1, \"name\": \"INVOKE\", \"code\": 0, \"data\": {\"count\": 0";
    prefix += ", \"algo_tick\": [[";
    prefix += std::to_string(algo_tick_us);
    prefix += "]]";
    prefix += ", \"boxes\": []";
    prefix += ", \"resolution\": [";
    prefix += std::to_string(width);
    prefix += ", ";
    prefix += std::to_string(height);
    prefix += "]";
    prefix += ", \"image\": \"";

    uart_send(prefix);
    uart_send_base64_stream(jpeg_data, jpeg_size);
    uart_send_literal("\"}}\n");
}

void viz_uart_send_raw_jpeg(const uint8_t *jpeg_data,
                           size_t jpeg_size,
                           uint16_t width,
                           uint16_t height)
{
    if (jpeg_data == nullptr || jpeg_size == 0U || jpeg_size > 0xFFFFU) {
        return;
    }
    // 8-byte binary header: [0xAA][0x55][size_lo][size_hi][w_lo][w_hi][h_lo][h_hi]
    const uint16_t sz16 = (uint16_t)jpeg_size;
    const char hdr[8] = {
        (char)0xAAU, (char)0x55U,
        (char)(sz16 & 0xFF), (char)((sz16 >> 8) & 0xFF),
        (char)(width & 0xFF), (char)((width >> 8) & 0xFF),
        (char)(height & 0xFF), (char)((height >> 8) & 0xFF),
    };
    uart_send_bytes(hdr, 8);
    // Send raw JPEG binary payload directly
    uart_send_bytes((const char *)jpeg_data, jpeg_size);
}

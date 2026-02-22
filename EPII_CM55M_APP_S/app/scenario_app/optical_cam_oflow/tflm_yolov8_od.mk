override SCENARIO_APP_SUPPORT_LIST := $(APP_TYPE)
override SCENARIO_APP_SUPPORT_LIST += $(APP_TYPE)/app
override SCENARIO_APP_SUPPORT_LIST += $(APP_TYPE)/pipeline
override SCENARIO_APP_SUPPORT_LIST += $(APP_TYPE)/io
override SCENARIO_APP_SUPPORT_LIST += $(APP_TYPE)/io/camera
override SCENARIO_APP_SUPPORT_LIST += $(APP_TYPE)/perf
override SCENARIO_APP_SUPPORT_LIST += $(APP_TYPE)/debug
override SCENARIO_APP_SUPPORT_LIST += $(APP_TYPE)/core
override SCENARIO_APP_SUPPORT_LIST += $(APP_TYPE)/config
override SCENARIO_APP_SUPPORT_LIST += $(APP_TYPE)/port

APPL_DEFINES += -DTFLM_YOLOV8_OD
APPL_DEFINES += -Daudvidpre_ret_pll400_timer1 -DIP_xdma
APPL_DEFINES += -DEVT_DATAPATH

# 保留串口日志输出。
APPL_DEFINES += -DUART_SEND_ALOGO_RESEULT
APPL_DEFINES += -DDBG_MORE

# 摄像头输入需要事件模块定义与数据路径配置。
EVENTHANDLER_SUPPORT = event_handler
EVENTHANDLER_SUPPORT_LIST += evt_datapath

##
# library support feature
# Add new library here
# The source code should be loacted in ~\library\{lib_name}\
##
# LIB_SEL = pwrmgmt sensordp tflmtag2209_u55tag2205 spi_ptl spi_eeprom hxevent img_proc
LIB_SEL = pwrmgmt sensordp tflmtag2412_u55tag2411 spi_ptl spi_eeprom hxevent img_proc

##
# middleware support feature
# Add new middleware here
# The source code should be loacted in ~\middleware\{mid_name}\
## 启用 FatFS，提供 ff.h/f_mount 等
MID_SEL = fatfs

# FatFS 端口选择，使用 SPI 卡
FATFS_PORT_LIST = mmc_spi

# 使能 CMSIS SPI 驱动，提供 Driver_SPI0 符号
CMSIS_DRIVERS_LIST = SPI

override OS_SEL:=
override TRUSTZONE := y
override TRUSTZONE_TYPE := security
override TRUSTZONE_FW_TYPE := 1
override CIS_SEL := HM_COMMON
override EPII_USECASE_SEL := drv_user_defined

CIS_SUPPORT_INAPP = cis_sensor
CIS_SUPPORT_INAPP_MODEL = cis_ov5647

ifeq ($(strip $(TOOLCHAIN)), arm)
override LINKER_SCRIPT_FILE := $(SCENARIO_APP_ROOT)/$(APP_TYPE)/TFLM_yolov8_od_S_only.sct
else#TOOLChain
override LINKER_SCRIPT_FILE := $(SCENARIO_APP_ROOT)/$(APP_TYPE)/TFLM_yolov8_od_S_only.ld
endif

##
# Add new external device here
# The source code should be located in ~\external\{device_name}\
##
#EXT_DEV_LIST += 

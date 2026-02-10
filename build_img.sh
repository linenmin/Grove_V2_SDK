#!/bin/bash
set -e

# 切换到工程根目录（脚本放在根目录时可省略）
# cd ~/Seeed_Grove_Vision_AI_Module_V2

echo "[1/4] Building firmware (quiet, errors still显示)..."
cd EPII_CM55M_APP_S

# 如需干净构建，取消下一行注释
# make clean

# -s 静默命令行；--no-print-directory 去掉递归提示
# 将 stdout 丢弃，只保留 stderr（警告/错误）。若需查看完整输出，去掉重定向。
make -s --no-print-directory -j4 >/dev/null
echo "Firmware build done."

echo "[2/4] Generating image..."
cd ../we2_image_gen_local
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/
./we2_local_image_gen project_case1_blp_wlcsp.json

echo "[3/4] Copying image to Windows..."
cp output_case1_sec_wlcsp/output.img \
   /mnt/d/BaiduNetdiskWorkspace/Leuven/AI_Master_Thesis/deployment/flash/img/output.img

echo "[4/4] Done! Image ready to flash."

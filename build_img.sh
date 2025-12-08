#!/bin/bash
set -e

# 切换到工程根目录（脚本放在根目录时可省略）
# cd ~/Seeed_Grove_Vision_AI_Module_V2

echo "[1/4] Building firmware..."
cd EPII_CM55M_APP_S

# 如果你需要clean，取消下面注释
# make clean

make -j4
echo "Firmware build done."

echo "[2/4] Generating image..."
cd ../we2_image_gen_local
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/
./we2_local_image_gen project_case1_blp_wlcsp.json

echo "[3/4] Copying image to Windows..."
cp output_case1_sec_wlcsp/output.img \
   /mnt/d/BaiduNetdiskWorkspace/Leuven/AI_Master_Thesis/deployment/flash/img/output.img

echo "[4/4] Done! Image ready to flash."

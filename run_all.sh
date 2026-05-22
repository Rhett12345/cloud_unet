#!/bin/bash
pgrep -f "main.py.*train" && echo "Training already running!" && exit 1
set -e   # 任意命令报错立即退出
#python main.py --stages fuse --split train --workers 8
#python main.py --stages fuse --split val --workers 8
#python main.py --stages fuse --split test --workers 8
#python main.py --stages stats
python main.py --stages train --resume /data/Data_yuq/unet_workdir/model/AGRI_GPM_Precip_UNet_best_csi.pth
#python main.py --stages train
python main.py --stages test
echo "All stages finished."
#python main.py --stages infer --agri_file <path>

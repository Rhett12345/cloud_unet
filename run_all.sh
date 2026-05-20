#!/bin/bash
set -e   # 任意命令报错立即退出
conda run -n cloudunet python main.py --stages fuse --split train --workers 8
conda run -n cloudunet python main.py --stages fuse --split val --workers 8
conda run -n cloudunet python main.py --stages fuse --split test --workers 8
conda run -n cloudunet python main.py --stages stats
conda run -n cloudunet python main.py --stages train
conda run -n cloudunet python main.py --stages test
echo "All stages finished."
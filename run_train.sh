#!/bin/bash

# 定义 Python 解释器的绝对路径 (根据之前的上下文)
PYTHON_EXEC="/home/yangyi/conda/envs/lights/bin/python"

# 定义项目路径
PROJECT_ROOT="/home/yangyi/lights/traffic_light_project"
SRC_DIR="${PROJECT_ROOT}/src"
DATA_ROOT="${PROJECT_ROOT}/data"
PRETRAINED_PATH="${PROJECT_ROOT}/weights/resnet18_a1_0-d63eafa0.pth"
LOG_FILE="${PROJECT_ROOT}/train.log"

# 进入源码目录
cd $SRC_DIR

# 使用 nohup 后台运行
echo "Starting training..."
nohup $PYTHON_EXEC train.py \
    --data_root $DATA_ROOT \
    --epochs 100 \
    --batch_size 32 \
    --gpus 0,1,2,3 \
    --pretrained_path $PRETRAINED_PATH \
    > $LOG_FILE 2>&1 &

# 获取 PID 2580137
PID=$!
echo "Training started in background with PID: $PID"
echo "Logs are being written to: $LOG_FILE"
echo "You can watch the logs using: tail -f $LOG_FILE"

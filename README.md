# Traffic Light Recognition with ViT-based Detector

本项目实现了基于 Vision Transformer (ViT) 骨干网络和 YOLO 检测头的交通信号灯检测模型。项目旨在探索 Transformer 在目标检测任务中的应用，特别是结合 YOLO 高效检测头的设计。

## 项目结构

```
traffic_light_project/
├── data/                 # 数据集目录
│   ├── test/             # 测试集图片
│   ├── train/            # 训练集 (images/ 和 labels/)
│   └── val/              # 验证集 (images/ 和 labels/)
├── src/                  # 源代码目录
│   ├── dataset.py        # 数据集加载 (YOLODataset)
│   ├── loss_yolo.py      # YOLO 损失函数定义
│   ├── model_yolo.py     # ViT-YOLO 模型定义
│   ├── prepare_data.py   # 数据清洗与划分脚本
│   ├── train.py          # 模型训练脚本
│   ├── utils.py          # 通用工具函数
│   └── visualize_test.py # 推理与可视化脚本
├── weights/              # 存放模型权重
├── results/              # 存放训练结果
├── requirements.txt      # 项目依赖
├── run_train.sh          # 训练启动脚本 (Shell)
└── README.md             # 项目说明文档
```

## 资源下载

本项目的相关资源已上传至 HuggingFace，请在开始前下载：

*   **数据集**: [traffic_light_data](https://huggingface.co/datasets/YYdream/traffic_light_data)
*   **预训练权重**: [traffic_light_weights](https://huggingface.co/YYdream/traffic_light_weights)

## 环境配置

1.  **安装依赖**:
    建议使用 Conda 创建虚拟环境，然后安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 数据准备

在开始训练之前，需要对原始数据进行清洗和划分。

1.  **准备原始数据**:
    将原始图片放入 `data/images`，对应的 YOLO 格式标签放入 `data/labels`。

2.  **运行预处理脚本**:
    该脚本会移除标签为空的无效数据，并按默认 8:2 的比例将数据划分为训练集 (`data/train`) 和验证集 (`data/val`)。
    ```bash
    python src/prepare_data.py
    ```
    *注意*: 如果 `data/train` 和 `data/val` 目录已存在，脚本可能会跳过划分步骤或报错，请确保在原始数据准备好后运行一次。

## 模型训练

### 1. 使用 Shell 脚本 (推荐)
项目提供了一个 `run_train.sh` 脚本，配置了常用的训练参数并支持后台运行。
请根据你的环境修改脚本中的 `PYTHON_EXEC` (Python解释器路径) 和 `PROJECT_ROOT` (项目根目录路径)。

```bash
bash run_train.sh
```

### 2. 手动运行 Python 命令
你也可以直接运行 `src/train.py`，并根据需要调整参数：

```bash
python src/train.py \
    --data_root ./data \
    --epochs 100 \
    --batch_size 16 \
    --device cuda \
    --pretrained_path weights/resnet18_pretrained.pth
```

**主要参数说明**:
*   `--data_root`: 数据集根目录 (默认为 `./data`)
*   `--epochs`: 训练轮数
*   `--batch_size`: 批次大小
*   `--device`: 使用设备 (`cuda` 或 `cpu`)
*   `--gpus`: 指定 GPU ID (例如 `0` 或 `0,1`)
*   `--pretrained_path`: 预训练权重路径 (可选)

## 推理与可视化

训练完成后，可以使用 `src/visualize_test.py` 加载模型权重，并在验证集或测试集上进行推理可视化。

```bash
python src/visualize_test.py \
    --checkpoint weights/checkpoint_best.pth \
    --split val \
    --device cuda
```

**参数说明**:
*   `--checkpoint`: 模型权重文件路径 (必须指定，或确保目录下有 `checkpoint_epoch_*.pth`)
*   `--split`: 数据集划分，可选 `val` (验证集) 或 `test` (测试集)
*   `--device`: 推理设备

## 核心文件说明 (`src/`)

*   **`model_yolo.py`**: 定义了 `ViTYOLO` 类。模型使用 `timm` 库加载 Vision Transformer 作为特征提取器，并连接自定义的 YOLO Head 进行目标检测。
*   **`loss_yolo.py`**: 实现了 YOLO 的损失计算，包含坐标回归损失 (MSE)、置信度损失 (BCE) 和分类损失 (CrossEntropy)。
*   **`dataset.py`**: 定义了 `YOLODataset`，负责加载图像和标签，并进行必要的数据增强（如亮度、对比度调整）。
*   **`train.py`**: 训练主程序，包含数据加载、模型初始化、训练循环和模型保存逻辑。


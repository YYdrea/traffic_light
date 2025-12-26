# Traffic Light Recognition with ViT-based Detector

本项目实现了基于 Vision Transformer (ViT) 的交通信号灯检测模型。项目中包含两种架构的尝试：基于 DETR 的简化版本和基于 YOLO head 的版本。目前的训练流程主要基于 YOLO 架构。

四张V100训练13个小时


## 项目结构

```
traffic_light_project/
  data/
    images/       # 原始图片文件夹
    labels/       # 原始 YOLO 格式标签文件夹
    train/        # 划分后的训练集
    val/          # 划分后的验证集
  src/
    ...           # 源代码文件
  weights/        # 存放训练好的模型权重
  results/        # 存放训练结果
  requirements.txt # 项目依赖
  run_train.sh    # 训练启动脚本
```

## 文件功能详解 (`src/` 目录)

以下是 `src/` 目录下各个文件的详细功能说明：

### 数据处理
*   **`dataset.py`**
    *   **功能**: 定义了 `YOLODataset` 类，继承自 `torch.utils.data.Dataset`。
    *   **作用**: 负责读取图像文件和对应的 `.txt` 标签文件。它将图像转换为 Tensor，并处理标签格式，以便 `DataLoader` 加载。
*   **`prepare_data.py`**
    *   **功能**: 数据预处理脚本。
    *   **作用**: 
        1.  **清洗数据**: 检查 `data/images` 和 `data/labels`，移除没有对应标签或标签为空的图片。
        2.  **划分数据集**: 将数据按默认 8:2 的比例划分为训练集 (`data/train`) 和验证集 (`data/val`)。
    *   **使用**: 在开始训练前运行一次。

### 模型定义
*   **`model_yolo.py`**
    *   **功能**: 定义了 `ViTYOLO` 模型。
    *   **作用**: 这是当前训练脚本主要使用的模型。它使用 `timm` 库加载 Vision Transformer (ViT) 作为骨干网络提取特征，并连接一个自定义的 `YOLOHead` 进行边界框回归和类别分类。
*   **`model.py`**
    *   **功能**: 定义了 `ViTDet` 模型。
    *   **作用**: 这是一个基于 DETR (Detection Transformer) 架构的实现，包含 ViT Backbone 和 Transformer Decoder。
    *   *注意*: 目前的 `train.py` 默认使用 `model_yolo.py`，此文件可能用于对比实验。

### 损失函数
*   **`loss_yolo.py`**
    *   **功能**: 定义了 `YOLOLoss` 类。
    *   **作用**: 实现了适用于 YOLO 架构的损失计算，包括：
        *   坐标损失 (MSE Loss)
        *   置信度损失 (BCE Loss)
        *   分类损失 (CrossEntropy Loss)
*   **`loss.py`**
    *   **功能**: 定义了 `HungarianMatcher` 和相关损失。
    *   **作用**: 用于 `ViTDet` 模型的二分图匹配损失（集合预测损失）。

### 训练与推理
*   **`train.py`**
    *   **功能**: 主训练脚本。
    *   **作用**: 
        *   解析命令行参数（Epochs, Batch Size, LR 等）。
        *   初始化数据加载器、模型 (`ViTYOLO`) 和优化器。
        *   执行训练循环，计算 Loss 并更新权重。
        *   定期保存模型权重到 `weights/` 目录。
*   **`visualize_test.py`**
    *   **功能**: 测试集可视化脚本。
    *   **作用**: 加载训练好的权重，对验证集图片进行推理，并将预测的边界框绘制在图片上，用于直观评估模型效果。
*   **`utils.py`**
    *   **功能**: 通用工具函数。
    *   **作用**: 包含坐标转换（如 `cxcywh` 转 `xyxy`）、IoU 计算 (`generalized_box_iou`) 等辅助函数。

## 使用方式

### 1. 环境配置
安装项目所需的 Python 依赖库：
```bash
pip install -r requirements.txt
```

### 2. 数据准备
确保原始数据位于 `data/images` 和 `data/labels` 中，然后运行数据准备脚本：
```bash
python src/prepare_data.py
```

### 3. 开始训练
推荐使用提供的 Shell 脚本进行训练，它已经配置好了常用的路径和参数：
```bash
bash run_train.sh
```
或者手动运行 Python 命令：
```bash
python src/train.py --epochs 100 --batch_size 16 --device cuda
```

### 4. 验证与可视化
训练完成后，可以使用以下命令查看模型在验证集上的表现：
```bash
python src/visualize_val.py --checkpoint weights/checkpoint_best.pth
```


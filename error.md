# 项目开发过程中的问题与解决方案总结

在本项目（基于 ViT 的交通信号灯检测）的开发过程中，我们遇到了多个工程和算法层面的问题。以下是详细的错误记录及解决方案。

## 1. 环境与依赖问题

### 1.1 无法安装 `albumentations` 库
*   **问题描述**: 开发环境中无法下载或安装 `albumentations` 图像增强库。
*   **解决方案**:
    *   移除对 `albumentations` 的依赖。
    *   在 `src/dataset.py` 中改用 `opencv` (`cv2.resize`) 或 `torchvision` 进行基础的图像预处理和缩放。

### 1.2 无法连接 Hugging Face 下载模型 (`timm`)
*   **问题描述**: 由于网络限制，`timm.create_model(..., pretrained=True)` 无法自动下载 ViT 预训练权重，报错 `MaxRetryError`。
*   **解决方案**:
    *   **方案一（临时）**: 曾尝试替换为 `torchvision.models.resnet18` 以规避 ViT 下载问题。
    *   **方案二（最终）**: 恢复使用 `timm`，但设置 `pretrained=False`。
    *   支持加载本地预训练权重：在 `src/train.py` 中添加 `--pretrained_path` 参数，允许用户手动加载下载好的权重文件。

## 2. 数据处理问题

### 2.1 输入尺寸不匹配 (Positional Embedding Mismatch)
*   **问题描述**: `RuntimeError: The size of tensor a (3601) must match the size of tensor b (197)`.
*   **原因**: ViT 模型需要固定的输入尺寸（如 224x224），因为其 Patch Embedding 和 Positional Embedding 的数量是固定的。原始图像尺寸不一致导致 Patch 数量与模型定义不符。
*   **解决方案**:
    *   在 `src/dataset.py` 和 `src/inference.py` 中，强制将输入图像 Resize 到固定尺寸（初期为 224x224，后期调整为 640x640）。

### 2.2 空标签导致的维度错误
*   **问题描述**: `RuntimeError: cdist only supports at least 2D tensors, X2 got: 1D`.
*   **原因**: 当某个 Batch 中的图片没有任何目标（没有红绿灯）时，`boxes` 张量变为空的一维张量，导致计算 Loss 矩阵时出错。
*   **解决方案**:
    *   在 `src/dataset.py` 中增加判断：如果 `boxes` 为空，强制返回一个形状为 `(0, 4)` 的二维张量。

### 2.3 无效标注框导致的断言错误
*   **问题描述**: `AssertionError` in `generalized_box_iou`.
*   **原因**: 数据集中存在宽度或高度 $\le 0$ 的无效标注框（例如 `x_max < x_min`）。
*   **解决方案**:
    *   在 `src/dataset.py` 加载标签时增加过滤逻辑，丢弃宽或高非正数的无效框。

### 2.4 训练集与验证集划分
*   **问题描述**: 原始数据混杂在一起，且包含大量空标签图片，影响训练效果且无法评估。
*   **解决方案**:
    *   编写 `src/prepare_data.py` 脚本。
    *   清洗数据：删除空标签的图片。
    *   划分数据：按 8:2 比例随机划分为 `train` 和 `val` 子集。

## 3. 模型训练问题

### 3.1 Loss 出现 NaN
*   **问题描述**: `Epoch 1 Average Loss: nan`.
*   **原因**:
    1.  当 Batch 中没有目标时，归一化因子 `num_boxes` 为 0，导致除零错误。
    2.  IoU 计算中分母可能为 0。
*   **解决方案**:
    *   在 `src/loss.py` 中，确保 `num_boxes` 至少为 1。
    *   在 `src/utils.py` 的 IoU 计算中，分母加上极小值 `1e-6` (`epsilon`)。

### 3.2 更改分辨率后的权重加载问题
*   **问题描述**: `RuntimeError: size mismatch for pos_embed`.
*   **原因**: 用户希望使用 640x640 分辨率，但预训练权重是基于 224x224 训练的。直接加载会导致位置编码形状不匹配（197 vs 1601）。
*   **解决方案**:
    *   在 `src/model.py` 中重写 `load_pretrained` 方法。
    *   手动实现**双三次插值 (Bicubic Interpolation)**，将预训练权重中的 `pos_embed` 调整为适应 640x640 输入的大小，然后再加载到模型中。

## 4. 架构调整与多卡训练问题 (ViT + YOLO Head)

### 4.1 DataParallel 属性访问错误
*   **问题描述**: `AttributeError: 'DataParallel' object has no attribute 'anchors'`.
*   **原因**: 在多卡训练时使用 `torch.nn.DataParallel` 包裹模型，模型的自定义属性（如 `anchors`）会被移动到 `model.module` 下，直接访问 `model.anchors` 会失败。
*   **解决方案**:
    *   在 `src/train.py` 中，在包裹 `DataParallel` 之前，先将 `anchors` 提取出来保存到变量中，再传给 Loss 函数。

### 4.2 CUDA Error: device-side assert triggered (类别数不匹配)
*   **问题描述**: 训练过程中报错 `CUDA error: device-side assert triggered`，通常发生在 Loss 计算阶段。
*   **原因**: 数据集中的标签包含类别 ID `3`，但训练脚本默认配置 `num_classes=3`（仅支持 ID 0, 1, 2）。在计算交叉熵损失时，标签索引越界导致 CUDA 断言失败。
*   **解决方案**:
    *   检查数据集标签，确认最大类别 ID。
    *   在 `src/train.py` 中将默认 `num_classes` 修改为 **4**（或根据实际数据调整），以覆盖所有出现的类别 ID。

## 5. 技术路线变更：从 ViTDet 到 ViT-YOLO

### 5.1 变更背景与原因
*   **初始方案 (ViTDet)**: 使用 ViT 提取特征，配合 Transformer Decoder 和 Hungarian Matcher 进行集合预测。
*   **遇到的问题**:
    *   **收敛困难**: 模型训练初期 Loss 下降缓慢，预测框难以匹配真实目标。
    *   **检测效果差**: 可视化结果显示大量漏检（无框）或置信度极低。
    *   **小目标挑战**: 纯 Transformer 架构在没有多尺度特征融合（FPN）的情况下，对交通信号灯这类小目标检测效果不理想。

### 5.2 新方案 (ViT-YOLO)
*   **架构调整**:
    *   **Backbone**: 保留 `ViT` (`timm`) 作为特征提取器。
    *   **Head**: 移除 Transformer Decoder，替换为 **YOLO Detection Head**（卷积层预测）。
    *   **Loss**: 移除 Hungarian Loss，替换为 **YOLO Loss** (MSE + BCE + CrossEntropy)。
    *   **Anchor**: 引入 Anchor Box 机制，利用先验框引导模型回归。
*   **预期优势**: YOLO 架构归纳偏置强，训练更稳定，且对小目标检测通常有更好的基线表现。

## 6. 最终技术路线变更：从 ViT-YOLO 到 ResNet-YOLO (Multi-Scale)

### 6.1 变更背景与原因
*   **初始选择 ViT 的理由**:
    *   **探索性**: 希望利用 Transformer 的全局注意力机制（Global Attention）来捕捉交通路口的全局上下文信息。
    *   **前沿性**: 验证 Vision Transformer 在传统目标检测任务上的潜力。
*   **遇到的问题**:
    *   **小目标丢失**: 即使切换到 YOLO Head，ViT-Tiny 的 Patch 机制（16x16）导致特征图分辨率较低，且缺乏 CNN 的层次化局部特征提取能力，对极小的交通信号灯检测效果依然不佳。
    *   **单尺度限制**: 最初的 ResNet-YOLO 仅使用了最后一层特征图 (Stride 32)，对于小目标来说，特征图上的一个点对应原图 32x32 的区域，信息丢失严重，导致重叠框多且定位不准。
    *   **训练震荡**: 纯 ViT 骨干在小数据集上训练较难收敛，容易过拟合或产生大量误检。

### 6.2 最终方案 (ResNet-YOLO Multi-Scale)
*   **架构调整**:
    *   **Backbone**: 移除 `ViT`，替换为经典的 **ResNet18**。
    *   **多尺度输出 (FPN思想)**: 借鉴 YOLOv3/U-Net 的思路，不再只输出最后一层，而是提取 **Stride 8 (P3), Stride 16 (P4), Stride 32 (P5)** 三个尺度的特征图。
    *   **多尺度 Head**: 为每个尺度设计独立的 YOLO Head，分别负责检测小、中、大目标。
    *   **多尺度 Anchors**: 针对不同尺度设计不同大小的 Anchor Box，Stride 8 负责极小目标（如远处的红绿灯）。
*   **优化措施**:
    *   **混合精度训练 (AMP)**: 引入 `torch.cuda.amp` 加速训练并减少显存占用。
    *   **数据增强**: 增加 `ColorJitter` 等增强手段防止过拟合。
    *   **Loss 权重调整**: 调高坐标回归权重 (`lambda_coord`)，调低无目标置信度权重 (`lambda_noobj`) 以抑制误检。

---
**最终总结**: 项目经历了 **ViTDet -> ViT-YOLO -> ResNet-YOLO (Single Scale) -> ResNet-YOLO (Multi-Scale)** 的演进。最终确认在当前数据量和任务特性（小目标检测）下，**多尺度 ResNet + YOLO** 是最稳定、高效的解决方案。

---
**总结**: 通过解决上述环境、数据和算法适配问题，项目成功实现了基于 ViT 的自定义目标检测流程，支持本地权重加载和自定义分辨率训练。

322752
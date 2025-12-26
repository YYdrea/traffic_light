import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import os

class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.output_dim = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, self.output_dim, 1)
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        output = self.conv(x)
        # [B, Anchors*(5+Classes), H, W] -> [B, Anchors, 5+Classes, H, W] -> [B, Anchors, H, W, 5+Classes]
        output = output.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        output = output.permute(0, 1, 3, 4, 2)
        return output

class ViTYOLO(nn.Module):
    def __init__(self, num_classes, img_size=640, backbone_name='resnet18', pretrained_path=None, pretrained=True):
        super().__init__()
        
        # 1. Backbone (Encoder) using timm
        # Using ResNet18
        # features_only=True returns a list of feature maps
        # out_indices=[2, 3, 4] means we take features at stride 8, 16, 32
        
        use_pretrained = pretrained and (pretrained_path is None)
        self.backbone = timm.create_model(backbone_name, pretrained=use_pretrained, features_only=True, out_indices=[2, 3, 4])
        
        if pretrained_path:
            if os.path.exists(pretrained_path):
                print(f"Loading backbone weights from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"Backbone weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            else:
                print(f"Warning: Pretrained path {pretrained_path} does not exist. Using random init.")
        
        # Get channel counts
        dummy_input = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        # features[0]: stride 8, features[1]: stride 16, features[2]: stride 32
        c3, c4, c5 = features[0].shape[1], features[1].shape[1], features[2].shape[1]
        
        # 2. YOLO Heads
        # Anchors for each scale (Small, Medium, Large)
        # P3 (Stride 8): Small objects
        self.anchors_p3 = [[10, 13], [16, 30], [33, 23]]
        # P4 (Stride 16): Medium objects
        self.anchors_p4 = [[30, 61], [62, 45], [59, 119]]
        # P5 (Stride 32): Large objects
        self.anchors_p5 = [[116, 90], [156, 198], [373, 326]]
        
        self.anchors = [self.anchors_p3, self.anchors_p4, self.anchors_p5]
        self.num_anchors = 3
        self.num_classes = num_classes
        
        self.head_p3 = YOLOHead(c3, self.num_anchors, num_classes)
        self.head_p4 = YOLOHead(c4, self.num_anchors, num_classes)
        self.head_p5 = YOLOHead(c5, self.num_anchors, num_classes)
        
    def forward(self, x):
        # x: [B, 3, H, W]
        features = self.backbone(x)
        p3, p4, p5 = features[0], features[1], features[2]
        
        out_p3 = self.head_p3(p3) # Stride 8
        out_p4 = self.head_p4(p4) # Stride 16
        out_p5 = self.head_p5(p5) # Stride 32
        
        return [out_p3, out_p4, out_p5]

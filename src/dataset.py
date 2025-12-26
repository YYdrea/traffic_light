import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        label_file = os.path.splitext(img_file)[0] + '.txt'
        
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, label_file)

        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 640x640 for ViT
        image = cv2.resize(image, (640, 640))
        
        h, w, _ = image.shape

        # Load Labels
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    
                    # Keep as cx, cy, w, h (normalized) for DETR-like models
                    # Ensure w and h are positive
                    if bw > 0 and bh > 0:
                        boxes.append([cx, cy, bw, bh])
                        labels.append(cls_id)

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        labels = torch.tensor(labels, dtype=torch.long)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        if self.transform:
            # Transform should handle conversion to Tensor if needed
            # Expecting transform to take numpy array (H, W, C)
            image = self.transform(image)
            
            # If transform didn't convert to tensor, do it here
            if not isinstance(image, torch.Tensor):
                 image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        else:
            # Normalize image to 0-1 and CHW
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

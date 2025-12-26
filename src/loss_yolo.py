import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size=640):
        super().__init__()
        self.anchors = anchors # List of lists of anchors
        self.num_classes = num_classes
        self.img_size = img_size
        self.ignore_thresh = 0.5
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, predictions, targets):
        # predictions: List of [B, Anchors, GridY, GridX, 5+Classes]
        # targets: List of dicts
        
        total_loss = 0
        loss_dict = {'loss': 0, 'box_loss': 0, 'obj_loss': 0, 'cls_loss': 0}
        
        # Iterate over each scale
        for scale_idx, prediction in enumerate(predictions):
            anchors_scale = self.anchors[scale_idx]
            loss, components = self.compute_loss_for_scale(prediction, targets, anchors_scale)
            
            total_loss += loss
            loss_dict['loss'] += loss
            loss_dict['box_loss'] += components['box_loss']
            loss_dict['obj_loss'] += components['obj_loss']
            loss_dict['cls_loss'] += components['cls_loss']
            
        return loss_dict

    def compute_loss_for_scale(self, prediction, targets, anchors):
        device = prediction.device
        B, num_anchors, grid_h, grid_w, _ = prediction.shape
        stride = self.img_size / grid_h
        
        # Transform anchors to tensor
        scaled_anchors = torch.tensor(anchors, device=device) / stride
        
        # Build targets
        mask = torch.zeros(B, num_anchors, grid_h, grid_w, device=device)
        noobj_mask = torch.ones(B, num_anchors, grid_h, grid_w, device=device)
        tx = torch.zeros(B, num_anchors, grid_h, grid_w, device=device)
        ty = torch.zeros(B, num_anchors, grid_h, grid_w, device=device)
        tw = torch.zeros(B, num_anchors, grid_h, grid_w, device=device)
        th = torch.zeros(B, num_anchors, grid_h, grid_w, device=device)
        tcls = torch.zeros(B, num_anchors, grid_h, grid_w, dtype=torch.long, device=device)
        
        # Process each image in batch
        for b in range(B):
            t = targets[b] # {'boxes': [N, 4], 'labels': [N]}
            if len(t['boxes']) == 0:
                continue
                
            # Convert boxes to grid coordinates
            # t['boxes'] is [cx, cy, w, h] normalized
            gxs = t['boxes'][:, 0] * grid_w
            gys = t['boxes'][:, 1] * grid_h
            gws = t['boxes'][:, 2] * grid_w
            ghs = t['boxes'][:, 3] * grid_h
            
            # Get grid indices
            gis = gxs.long()
            gjs = gys.long()
            
            # Clamp indices to be within grid
            gis = gis.clamp(0, grid_w - 1)
            gjs = gjs.clamp(0, grid_h - 1)
            
            # Match anchors
            anchor_boxes = torch.zeros(len(anchors), 4, device=device)
            anchor_boxes[:, 2] = scaled_anchors[:, 0]
            anchor_boxes[:, 3] = scaled_anchors[:, 1]
            
            gt_boxes = torch.zeros(len(t['boxes']), 4, device=device)
            gt_boxes[:, 2] = gws
            gt_boxes[:, 3] = ghs
            
            # Calculate IoU (wh only)
            inter_w = torch.min(anchor_boxes[:, 2].unsqueeze(1), gt_boxes[:, 2].unsqueeze(0))
            inter_h = torch.min(anchor_boxes[:, 3].unsqueeze(1), gt_boxes[:, 3].unsqueeze(0))
            inter_area = (inter_w * inter_h).clamp(min=0)
            
            anchor_area = anchor_boxes[:, 2] * anchor_boxes[:, 3]
            gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
            
            union_area = anchor_area.unsqueeze(1) + gt_area.unsqueeze(0) - inter_area
            iou = inter_area / (union_area + 1e-6) # [Num_Anchors, N]
            
            # Best anchor for each target
            best_iou, best_anchor_idx = iou.max(0) # [N]
            
            # Assign targets
            for i, anchor_idx in enumerate(best_anchor_idx):
                gj, gi = gjs[i], gis[i]
                
                # Mask
                mask[b, anchor_idx, gj, gi] = 1
                noobj_mask[b, anchor_idx, gj, gi] = 0
                
                # Coordinates
                tx[b, anchor_idx, gj, gi] = gxs[i] - gi.float()
                ty[b, anchor_idx, gj, gi] = gys[i] - gj.float()
                
                tw[b, anchor_idx, gj, gi] = torch.log(gws[i] / scaled_anchors[anchor_idx, 0] + 1e-16)
                th[b, anchor_idx, gj, gi] = torch.log(ghs[i] / scaled_anchors[anchor_idx, 1] + 1e-16)
                
                # Class
                tcls[b, anchor_idx, gj, gi] = t['labels'][i]
                
                # Ignore high IoU anchors for noobj loss
                for a in range(len(anchors)):
                    if iou[a, i] > self.ignore_thresh:
                        noobj_mask[b, a, gj, gi] = 0

        # --- Compute Loss ---
        pred_x = torch.sigmoid(prediction[..., 0])
        pred_y = torch.sigmoid(prediction[..., 1])
        pred_w = prediction[..., 2]
        pred_h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = prediction[..., 5:]
        
        # 1. Coordinate Loss
        if mask.sum() > 0:
            loss_x = self.mse_loss(pred_x[mask==1], tx[mask==1])
            loss_y = self.mse_loss(pred_y[mask==1], ty[mask==1])
            loss_w = self.mse_loss(pred_w[mask==1], tw[mask==1])
            loss_h = self.mse_loss(pred_h[mask==1], th[mask==1])
            loss_coord = loss_x + loss_y + loss_w + loss_h
        else:
            loss_coord = torch.tensor(0.0, device=device)
        
        # 2. Objectness Loss
        if mask.sum() > 0:
            loss_conf_obj = self.bce_loss(pred_conf[mask==1], mask[mask==1])
        else:
            loss_conf_obj = torch.tensor(0.0, device=device)
            
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask==1], mask[noobj_mask==1])
        
        # Weights
        lambda_coord = 5.0
        lambda_noobj = 0.5
        lambda_obj = 5.0 # Increase positive objectness weight to reduce missed detections
        
        loss_conf = lambda_obj * loss_conf_obj + lambda_noobj * loss_conf_noobj
        
        # 3. Class Loss
        if mask.sum() > 0:
            # Reshape for CrossEntropyLoss: [N, C] vs [N]
            # pred_cls[mask==1] is [N, C]
            # tcls[mask==1] is [N]
            loss_cls = self.ce_loss(pred_cls[mask==1], tcls[mask==1])
        else:
            loss_cls = torch.tensor(0.0, device=device)
            
        loss = lambda_coord * loss_coord + loss_conf + loss_cls
        
        num_all_anchors = B * num_anchors * grid_h * grid_w
        return loss / num_all_anchors, {'box_loss': loss_coord.item() / num_all_anchors, 'obj_loss': loss_conf.item() / num_all_anchors, 'cls_loss': loss_cls.item() / num_all_anchors}

import torch
import cv2
import argparse
import numpy as np
import os
import random
import glob
from torchvision.ops import nms
from model_yolo import ViTYOLO

def visualize_val(args):
    device = torch.device(args.device)
    
    # 1. Find Checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try to find the latest checkpoint
        checkpoints = glob.glob("checkpoint_epoch_*.pth")
        if not checkpoints:
            print("Error: No checkpoint found. Please train the model first or specify --checkpoint.")
            return
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = checkpoints[-1]
        print(f"Using latest checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} does not exist.")
        return

    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    # We are loading a full checkpoint, so we don't need to download ImageNet weights
    model = ViTYOLO(num_classes=args.num_classes, img_size=640, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Select Random Images
    if args.split == 'test':
        img_dir = os.path.join(args.data_root, 'test')
        if not os.path.exists(img_dir):
            img_dir = os.path.join(args.data_root, 'test', 'images')
        label_dir = None
    else:
        img_dir = os.path.join(args.data_root, args.split, 'images')
        label_dir = os.path.join(args.data_root, args.split, 'labels')

    if not os.path.exists(img_dir):
        print(f"Error: Image directory {img_dir} does not exist.")
        return
        
    all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not all_images:
        print(f"No images found in {img_dir}.")
        return
        
    selected_images = random.sample(all_images, min(args.num_samples, len(all_images)))
    print(f"Selected {len(selected_images)} images for visualization from {img_dir}.")
    
    # 4. Create Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. Inference Loop
    for img_file in selected_images:
        img_path = os.path.join(img_dir, img_file)
        
        # Load Image
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        
        # Create a copy for Ground Truth
        image_gt = image_bgr.copy()
        
        # Draw Ground Truth
        if label_dir:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        
                        h, w, _ = image_gt.shape
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        
                        # Class names
                        class_names = {0: 'Red', 1: 'Green', 2: 'Yellow', 3: 'Off'}
                        class_name = class_names.get(cls_id, str(cls_id))
                        
                        # Draw rectangle (Blue for GT)
                        cv2.rectangle(image_gt, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(image_gt, f"GT: {class_name}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                cv2.putText(image_gt, "No Label File", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
             cv2.putText(image_gt, "No Labels Dir", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        
        # Preprocess
        img_resized = cv2.resize(image_rgb, (640, 640))
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            # outputs is now a list of tensors
            
        # Post-process YOLO
        all_pred_boxes = []
        all_pred_conf = []
        all_pred_cls_prob = []
        
        for scale_idx, output in enumerate(outputs):
            # output: [B, Anchors, GridY, GridX, 5+Classes]
            B, num_anchors, grid_h, grid_w, _ = output.shape
            stride = 640 / grid_h
            anchors = model.anchors[scale_idx]
            scaled_anchors = torch.tensor(anchors, device=device) / stride
            
            pred_x = torch.sigmoid(output[..., 0])
            pred_y = torch.sigmoid(output[..., 1])
            pred_w = output[..., 2]
            pred_h = output[..., 3]
            pred_conf = torch.sigmoid(output[..., 4])
            pred_cls = output[..., 5:] # Logits
            pred_cls_prob = torch.softmax(pred_cls, dim=-1)
            
            # Create grid
            grid_x = torch.arange(grid_w, device=device).repeat(grid_h, 1).view([1, 1, grid_h, grid_w])
            grid_y = torch.arange(grid_h, device=device).repeat(grid_w, 1).t().view([1, 1, grid_h, grid_w])
            
            anchor_w = scaled_anchors[:, 0].view(1, num_anchors, 1, 1)
            anchor_h = scaled_anchors[:, 1].view(1, num_anchors, 1, 1)
            
            # Decode
            pred_boxes = torch.zeros_like(output[..., :4])
            pred_boxes[..., 0] = (pred_x + grid_x) * stride
            pred_boxes[..., 1] = (pred_y + grid_y) * stride
            pred_boxes[..., 2] = torch.exp(pred_w) * anchor_w * stride
            pred_boxes[..., 3] = torch.exp(pred_h) * anchor_h * stride
            
            # Flatten
            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_conf = pred_conf.reshape(-1)
            pred_cls_prob = pred_cls_prob.reshape(-1, args.num_classes)
            
            # Filter by confidence
            mask = pred_conf > args.conf_thresh
            all_pred_boxes.append(pred_boxes[mask])
            all_pred_conf.append(pred_conf[mask])
            all_pred_cls_prob.append(pred_cls_prob[mask])
            
        # Concatenate all scales
        if len(all_pred_boxes) > 0:
            pred_boxes = torch.cat(all_pred_boxes, dim=0)
            pred_conf = torch.cat(all_pred_conf, dim=0)
            pred_cls_prob = torch.cat(all_pred_cls_prob, dim=0)
        else:
            pred_boxes = torch.tensor([], device=device)
            pred_conf = torch.tensor([], device=device)
            pred_cls_prob = torch.tensor([], device=device)
        
        if len(pred_boxes) == 0:
            print(f"Image: {img_file} - No detections above threshold.")
            combined_img = np.hstack((image_gt, image_bgr))
            save_path = os.path.join(args.output_dir, f"compare_{img_file}")
            cv2.imwrite(save_path, combined_img)
            continue
            
        # NMS
        # Convert cx, cy, w, h to x1, y1, x2, y2
        boxes_xyxy = torch.zeros_like(pred_boxes)
        boxes_xyxy[:, 0] = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        boxes_xyxy[:, 1] = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        boxes_xyxy[:, 2] = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        boxes_xyxy[:, 3] = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        # Get max class score
        class_scores, class_ids = pred_cls_prob.max(1)
        final_scores = pred_conf * class_scores
        
        # Filter by final score
        score_mask = final_scores > args.conf_thresh
        boxes_xyxy = boxes_xyxy[score_mask]
        final_scores = final_scores[score_mask]
        class_ids = class_ids[score_mask]
        
        # Apply NMS
        # Use torchvision.ops.nms
        # Lower IoU threshold to 0.2 to suppress more overlapping boxes
        keep = nms(boxes_xyxy, final_scores, 0.2)
        
        boxes_xyxy = boxes_xyxy[keep]
        final_scores = final_scores[keep]
        class_ids = class_ids[keep]
        
        # Scale to original image size
        scale_w = w / 640
        scale_h = h / 640
        
        boxes_xyxy[:, 0] *= scale_w
        boxes_xyxy[:, 2] *= scale_w
        boxes_xyxy[:, 1] *= scale_h
        boxes_xyxy[:, 3] *= scale_h
        
        print(f"Image: {img_file} - Found {len(boxes_xyxy)} objects.")
        
        # Class names
        class_names = {0: 'Red', 1: 'Green', 2: 'Yellow', 3: 'Off'}

        # Draw
        for box, score, cls_id in zip(boxes_xyxy, final_scores, class_ids):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            cls_id_int = int(cls_id.item())
            class_name = class_names.get(cls_id_int, str(cls_id_int))
            
            # Draw rectangle
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{class_name} ({score:.2f})"
            cv2.putText(image_bgr, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Concatenate images
        combined_img = np.hstack((image_gt, image_bgr))
        
        # Save result
        save_path = os.path.join(args.output_dir, f"compare_{img_file}")
        cv2.imwrite(save_path, combined_img)
        print(f"Saved result to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data', help='Root path to data')
    parser.add_argument('--checkpoint', type=str, default='/home/yangyi/lights/traffic_light_project/weights/checkpoint_best.pth', help='Path to checkpoint (optional, defaults to latest)')
    parser.add_argument('--output_dir', type=str, default='../results', help='Directory to save results')
    parser.add_argument('--split', type=str, default='val', help='Data split to visualize (val or test)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of random samples')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--conf_thresh', type=float, default=0.99) # Lower threshold for visualization
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    visualize_val(args)

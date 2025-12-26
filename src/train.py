import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

from dataset import YOLODataset, collate_fn
from model_yolo import ViTYOLO
from loss_yolo import YOLOLoss
from torchvision import transforms

def train(args):
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        gpu_ids = None
        device = torch.device(args.device)
    
    # 1. Dataset & Dataloader
    # Augmentation for training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(), # Converts to [0, 1]
    ])

    # Train set
    train_img_dir = os.path.join(args.data_root, 'train', 'images')
    train_label_dir = os.path.join(args.data_root, 'train', 'labels')
    train_dataset = YOLODataset(train_img_dir, train_label_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    # Val set
    val_img_dir = os.path.join(args.data_root, 'val', 'images')
    val_label_dir = os.path.join(args.data_root, 'val', 'labels')
    val_dataset = YOLODataset(val_img_dir, val_label_dir) # No transform for validation
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    print(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images.")
    
    # Enable CUDNN benchmark for constant input sizes
    torch.backends.cudnn.benchmark = True
    
    # 2. Model
    model = ViTYOLO(num_classes=args.num_classes, img_size=640, pretrained_path=args.pretrained_path)
    model.to(device)
    
    # Save anchors before wrapping in DataParallel
    anchors = model.anchors

    # Multi-GPU support
    if gpu_ids:
        print(f"Using GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    elif torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    # 3. Loss
    criterion = YOLOLoss(anchors=anchors, num_classes=args.num_classes, img_size=640)
    criterion.to(device)
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    # Define save directory
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights')
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        train_loss = 0
        
        for images, targets in pbar:
            images = torch.stack(images).to(device)
            # targets is a list of dicts, YOLOLoss expects this
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss_dict = criterion(outputs, targets)
            losses = loss_dict['loss']
            
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
            pbar.set_postfix({'loss': losses.item(), 'lr': optimizer.param_groups[0]['lr']})
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = torch.stack(images).to(device)
                
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                losses = loss_dict['loss']
                val_loss += losses.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(save_dir, "checkpoint_best.pth"))
            print(f"New best model saved with val loss {best_val_loss:.4f}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            # Handle DataParallel wrapper
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            
        # Always save latest
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, "checkpoint_latest.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/yangyi/lights/traffic_light_project/data', help='Root path to data containing train/ and val/ folders')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes (e.g. Red, Green, Yellow, Off)')
    parser.add_argument('--num_queries', type=int, default=100, help='Number of object queries')
    parser.add_argument('--pretrained_path', type=str, default='/home/yangyi/lights/traffic_light_project/weights/resnet18_pretrained.pth', help='Path to pretrained ViT weights')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='Comma separated list of GPU ids to use (e.g. "0,1,2,3")')
    
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: Train directory {train_dir} does not exist. Please run prepare_data.py first.")
    else:
        train(args)

#2979089
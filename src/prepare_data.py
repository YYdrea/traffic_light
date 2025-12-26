import os
import shutil
import random
from tqdm import tqdm

def clean_and_split(data_root, split_ratio=0.8):
    # 原始路径
    img_dir = os.path.join(data_root, 'images')
    label_dir = os.path.join(data_root, 'labels')
    
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"Error: {img_dir} or {label_dir} does not exist.")
        print("If you have already split the dataset, please skip this step.")
        return

    # 1. 清洗数据 (Cleaning)
    print("Cleaning dataset (removing empty labels)...")
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    removed_count = 0
    valid_files = []
    
    for img_file in tqdm(img_files):
        name = os.path.splitext(img_file)[0]
        label_file = name + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        is_empty = True
        if os.path.exists(label_path):
            # 检查文件大小或内容
            if os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        is_empty = False
        
        if is_empty:
            # 删除图片和标签
            img_path = os.path.join(img_dir, img_file)
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            removed_count += 1
        else:
            valid_files.append(img_file)
            
    print(f"Removed {removed_count} empty samples. Remaining: {len(valid_files)}")
    
    if len(valid_files) == 0:
        print("No valid files found!")
        return

    # 2. 划分数据集 (Splitting)
    print("Splitting dataset into train and val...")
    random.seed(42) # 保证每次划分一致
    random.shuffle(valid_files)
    
    split_idx = int(len(valid_files) * split_ratio)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]
    
    # 创建新的目录结构
    train_img_dir = os.path.join(data_root, 'train', 'images')
    train_label_dir = os.path.join(data_root, 'train', 'labels')
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_label_dir = os.path.join(data_root, 'val', 'labels')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    def move_files(files, dest_img_dir, dest_lbl_dir):
        for img_file in tqdm(files):
            # 移动图片
            src_img = os.path.join(img_dir, img_file)
            dst_img = os.path.join(dest_img_dir, img_file)
            shutil.move(src_img, dst_img)
            
            # 移动标签
            name = os.path.splitext(img_file)[0]
            label_file = name + '.txt'
            src_lbl = os.path.join(label_dir, label_file)
            dst_lbl = os.path.join(dest_lbl_dir, label_file)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

    print(f"Moving {len(train_files)} files to train set...")
    move_files(train_files, train_img_dir, train_label_dir)
    
    print(f"Moving {len(val_files)} files to val set...")
    move_files(val_files, val_img_dir, val_label_dir)
    
    # 尝试删除空的旧目录
    try:
        os.rmdir(img_dir)
        os.rmdir(label_dir)
    except:
        pass
        
    print("Done! Dataset structure is now:")
    print(f"  {train_img_dir}")
    print(f"  {val_img_dir}")

if __name__ == '__main__':
    # 假设脚本在 src 目录下运行，数据在 ../data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    clean_and_split(data_path)

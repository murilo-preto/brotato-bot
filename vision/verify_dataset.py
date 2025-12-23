import os
import cv2
import random
from pathlib import Path

def verify_yolo_dataset(dataset_path):
    """Verify YOLO dataset structure and visualize samples"""
    
    images_train_dir = os.path.join(dataset_path, 'images', 'train')
    labels_train_dir = os.path.join(dataset_path, 'labels', 'train')
    
    # Get list of images
    image_files = [f for f in os.listdir(images_train_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found!")
        return
    
    # Check a few random samples
    for _ in range(3):
        img_file = random.choice(image_files)
        img_path = os.path.join(images_train_dir, img_file)
        label_path = os.path.join(labels_train_dir, img_file.replace('.png', '.txt').replace('.jpg', '.txt'))
        
        print(f"\nVerifying: {img_file}")
        print(f"Image path: {img_path}")
        print(f"Label path: {label_path}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ❌ Failed to load image")
            continue
        
        height, width = img.shape[:2]
        print(f"  Image size: {width}x{height}")
        
        # Load labels
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            print(f"  Found {len(lines)} annotations")
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, w, h = map(float, parts)
                    # Convert back to pixel coordinates for verification
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)
                    print(f"    Class {int(class_id)}: [{x1}, {y1}, {x2}, {y2}]")
        else:
            print("  ⚠️ No label file found (might be empty image)")
    
    # Check dataset structure
    print(f"\n📁 Dataset structure:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

if __name__ == '__main__':
    verify_yolo_dataset('yolo_dataset')
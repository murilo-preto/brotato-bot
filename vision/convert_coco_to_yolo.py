import json
import os
import shutil
import argparse
import random
from pathlib import Path
from collections import Counter

def analyze_dataset(coco_data):
    
    total_images = len(coco_data['images'])
    annotated_image_ids = set(ann['image_id'] for ann in coco_data['annotations'])
    images_with_annotations = len(annotated_image_ids)
    images_without_annotations = total_images - images_with_annotations
    
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Images with annotations: {images_with_annotations} ({images_with_annotations/total_images*100:.1f}%)")
    print(f"Images without annotations: {images_without_annotations} ({images_without_annotations/total_images*100:.1f}%)")
    
    annotations_by_image = Counter([ann['image_id'] for ann in coco_data['annotations']])
    print(f"\nAnnotation distribution:")
    for count in sorted(Counter(annotations_by_image.values()).items()):
        print(f"  {count[0]} annotations: {count[1]} images")
    
    class_counts = Counter([ann['category_id'] for ann in coco_data['annotations']])
    category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"\nClass distribution:")
    for class_id, count in class_counts.most_common():
        print(f"  {category_names[class_id]} (ID {class_id}): {count} instances")
    
    return annotated_image_ids, images_without_annotations

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, 
                         split_ratio=0.8, filter_unlabeled=True, 
                         include_empty_as_negative=False):
    
    print("Starting COCO to YOLO conversion...")
    print(f"COCO JSON: {coco_json_path}")
    print(f"Images dir: {images_dir}")
    print(f"Output dir: {output_dir}")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    annotated_image_ids, images_without_annotations = analyze_dataset(coco_data)
    
    if filter_unlabeled and images_without_annotations > 0:
        if include_empty_as_negative:
            print(f"\nIncluding {images_without_annotations} unlabeled images as negative samples")
        else:
            print(f"\nFiltering out {images_without_annotations} unlabeled images")
            coco_data['images'] = [img for img in coco_data['images'] 
                                   if img['id'] in annotated_image_ids]
            print(f"Keeping {len(coco_data['images'])} labeled images")
    else:
        include_empty_as_negative = True
    
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    category_id_to_class = {cat['id']: idx for idx, cat in enumerate(categories)}
    class_id_to_name = {idx: cat['name'] for idx, cat in enumerate(categories)}
    
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    output_dirs = {
        'images': {
            'train': os.path.join(output_dir, 'images', 'train'),
            'val': os.path.join(output_dir, 'images', 'val')
        },
        'labels': {
            'train': os.path.join(output_dir, 'labels', 'train'),
            'val': os.path.join(output_dir, 'labels', 'val')
        }
    }
    
    for dir_type in output_dirs.values():
        for split_dir in dir_type.values():
            os.makedirs(split_dir, exist_ok=True)
    
    all_image_ids = list(image_id_to_info.keys())
    random.shuffle(all_image_ids)
    split_idx = int(len(all_image_ids) * split_ratio)
    train_ids = all_image_ids[:split_idx]
    val_ids = all_image_ids[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Training images: {len(train_ids)}")
    print(f"  Validation images: {len(val_ids)}")
    print(f"  Split ratio: {split_ratio}")
    
    print(f"\nCategory mapping:")
    for idx, cat in enumerate(categories):
        print(f"  Class {idx}: {cat['name']} (COCO ID: {cat['id']})")
    
    stats = {
        'train': {'images': 0, 'objects': 0, 'empty': 0},
        'val': {'images': 0, 'objects': 0, 'empty': 0}
    }
    
    for split_name, image_ids in [('train', train_ids), ('val', val_ids)]:
        print(f"\nProcessing {split_name} set...")
        
        for img_id in image_ids:
            img_info = image_id_to_info[img_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            src_img_path = os.path.join(images_dir, img_filename)
            
            if not os.path.exists(src_img_path):
                print(f"  Warning: Image not found: {src_img_path}")
                continue
            
            dest_img_path = os.path.join(output_dirs['images'][split_name], img_filename)
            dest_label_path = os.path.join(
                output_dirs['labels'][split_name], 
                img_filename.replace('.png', '.txt').replace('.jpg', '.txt')
            )
            
            shutil.copy2(src_img_path, dest_img_path)
            stats[split_name]['images'] += 1
            
            with open(dest_label_path, 'w') as f:
                if img_id in annotations_by_image and annotations_by_image[img_id]:
                    object_count = 0
                    for ann in annotations_by_image[img_id]:
                        x_min, y_min, bbox_width, bbox_height = ann['bbox']
                        
                        if bbox_width <= 0 or bbox_height <= 0:
                            continue
                        
                        x_center = (x_min + bbox_width / 2) / img_width
                        y_center = (y_min + bbox_height / 2) / img_height
                        width_norm = bbox_width / img_width
                        height_norm = bbox_height / img_height
                        
                        if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 <= width_norm <= 1 and 0 <= height_norm <= 1):
                            
                            class_id = category_id_to_class[ann['category_id']]
                            
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                            object_count += 1
                    
                    stats[split_name]['objects'] += object_count
                    
                    if object_count == 0:
                        stats[split_name]['empty'] += 1
                else:
                    stats[split_name]['empty'] += 1
    
    print(f"\nConversion completed!")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    print(f"\nFinal Statistics:")
    print(f"{'Split':<10} {'Images':<10} {'Objects':<10} {'Empty':<10} {'Obj/Img':<10}")
    print("-" * 50)
    for split in ['train', 'val']:
        images = stats[split]['images']
        objects = stats[split]['objects']
        empty = stats[split]['empty']
        obj_per_img = objects / (images - empty) if (images - empty) > 0 else 0
        print(f"{split:<10} {images:<10} {objects:<10} {empty:<10} {obj_per_img:.2f}")
    
    total_images = stats['train']['images'] + stats['val']['images']
    total_objects = stats['train']['objects'] + stats['val']['objects']
    print(f"\nTotals: {total_images} images, {total_objects} objects")
    
    yaml_path = create_dataset_yaml(output_dir, categories)
    
    print(f"\nDataset configuration saved to: {yaml_path}")
    print(f"\nDirectory structure:")
    print_output_structure(output_dir)
    
    return output_dir, yaml_path

def create_dataset_yaml(output_dir, categories):
    
    yaml_content = f"""# YOLO Dataset Configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test:

nc: {len(categories)}

names:
"""
    
    for idx, cat in enumerate(categories):
        yaml_content += f"  {idx}: {cat['name']}\n"
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def print_output_structure(output_dir):
    
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        
        if files:
            extensions = {}
            for file in files:
                ext = os.path.splitext(file)[1]
                extensions[ext] = extensions.get(ext, 0) + 1
            
            for ext, count in extensions.items():
                if count <= 5:
                    ext_files = [f for f in files if f.endswith(ext)]
                    for file in sorted(ext_files)[:5]:
                        print(f"{subindent}{file}")
                    if len(ext_files) > 5:
                        print(f"{subindent}... and {len(ext_files) - 5} more {ext} files")
                else:
                    print(f"{subindent}{count} {ext[1:]} files")

def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO format annotations to YOLO format with filtering options',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_coco_to_yolo.py --coco-json annotations.json --images-dir images
  
  python convert_coco_to_yolo.py --coco-json annotations.json --images-dir images --include-empty
  
  python convert_coco_to_yolo.py --coco-json annotations.json --images-dir images --output-dir my_dataset --split-ratio 0.9
  
  python convert_coco_to_yolo.py --coco-json annotations.json --images-dir images --no-filter
        """
    )
    
    parser.add_argument('--coco-json', type=str, required=True,
                       help='Path to COCO JSON annotation file')
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output-dir', type=str, default='yolo_dataset',
                       help='Output directory for YOLO format (default: yolo_dataset)')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                       help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--no-filter', action='store_false', dest='filter_unlabeled',
                       help='Do not filter unlabeled images (keep all)')
    parser.add_argument('--include-empty', action='store_true',
                       help='Include empty images as negative samples (creates empty label files)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splitting (default: 42)')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Configuration:")
    print(f"  COCO JSON: {args.coco_json}")
    print(f"  Images dir: {args.images_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Split ratio: {args.split_ratio}")
    print(f"  Filter unlabeled: {args.filter_unlabeled}")
    print(f"  Include empty as negatives: {args.include_empty}")
    print(f"  Random seed: {args.seed}")
    
    if not os.path.exists(args.coco_json):
        print(f"Error: COCO JSON file not found: {args.coco_json}")
        return
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        output_dir, yaml_path = convert_coco_to_yolo(
            coco_json_path=args.coco_json,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            split_ratio=args.split_ratio,
            filter_unlabeled=args.filter_unlabeled,
            include_empty_as_negative=args.include_empty
        )
        
        print(f"\nConversion successful!")
        print(f"\nNext steps:")
        print(f"1. Verify dataset: python verify_yolo_dataset.py --dataset-dir {output_dir}")
        print(f"2. Train YOLO model: python train_yolo.py --data {yaml_path}")
        print(f"3. For quick verification, check a few samples:")
        print(f"   - Images: {os.path.join(output_dir, 'images/train')}")
        print(f"   - Labels: {os.path.join(output_dir, 'labels/train')}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
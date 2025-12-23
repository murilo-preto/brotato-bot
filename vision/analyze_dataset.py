import json
import os
from collections import Counter
import cv2
import numpy as np

def analyze_coco_dataset(coco_json_path, images_dir):
    """
    Analyze COCO dataset to understand annotation coverage
    """
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Statistics
    total_images = len(coco_data['images'])
    annotations_by_image = Counter([ann['image_id'] for ann in coco_data['annotations']])
    
    images_with_annotations = len(annotations_by_image)
    images_without_annotations = total_images - images_with_annotations
    
    # Distribution of annotation counts
    annotation_counts = Counter(annotations_by_image.values())
    
    print("=" * 60)
    print("📊 DATASET ANALYSIS REPORT")
    print("=" * 60)
    print(f"\n📈 Basic Statistics:")
    print(f"  Total images: {total_images}")
    print(f"  Images with annotations: {images_with_annotations} ({images_with_annotations/total_images*100:.1f}%)")
    print(f"  Images without annotations: {images_without_annotations} ({images_without_annotations/total_images*100:.1f}%)")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Average annotations per image: {len(coco_data['annotations'])/images_with_annotations:.2f}")
    
    print(f"\n📊 Annotation Distribution:")
    for count in sorted(annotation_counts.keys()):
        print(f"  {count} annotations: {annotation_counts[count]} images")
    
    print(f"\n🏷️  Class Distribution:")
    class_counts = Counter([ann['category_id'] for ann in coco_data['annotations']])
    category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    for class_id, count in class_counts.most_common():
        print(f"  {category_names[class_id]} (ID {class_id}): {count} instances")
    
    # Check if unlabeled images are actually empty
    print(f"\n🔍 Checking Unlabeled Images:")
    unlabeled_image_ids = set(img['id'] for img in coco_data['images']) - set(annotations_by_image.keys())
    
    # Sample a few unlabeled images to check manually
    sample_size = min(5, len(unlabeled_image_ids))
    sample_unlabeled = list(unlabeled_image_ids)[:sample_size]
    
    print(f"  Sampling {sample_size} unlabeled images for visual inspection:")
    for img_id in sample_unlabeled:
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                print(f"    Image {img_info['file_name']}: Size {img.shape[1]}x{img.shape[0]}")
                # You could add object detection here to see if objects exist
            else:
                print(f"    Image {img_info['file_name']}: Cannot read")
        else:
            print(f"    Image {img_info['file_name']}: Not found")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if images_without_annotations == 0:
        print("  ✅ All images are labeled. Perfect dataset!")
    elif images_without_annotations / total_images < 0.1:
        print(f"  ⚠️  {images_without_annotations} unlabeled images ({images_without_annotations/total_images*100:.1f}%)")
        print("  Suggestion: Consider keeping them as negative samples if they're truly empty")
    else:
        print(f"  ❗ {images_without_annotations} unlabeled images ({images_without_annotations/total_images*100:.1f}%)")
        print("  Suggestion: Review these images and either:")
        print("    - Label them if they contain objects")
        print("    - Remove them if they're irrelevant")
        print("    - Keep as negative samples if they're truly empty")
    
    return {
        'total_images': total_images,
        'labeled_images': images_with_annotations,
        'unlabeled_images': images_without_annotations,
        'total_annotations': len(coco_data['annotations']),
        'class_distribution': dict(class_counts)
    }

if __name__ == '__main__':
    stats = analyze_coco_dataset(
        "video_0/annotations/instances_Train.json",
        "video_0/images/Train"
    )
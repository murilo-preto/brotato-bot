"""
Use trained YOLO model to auto-annotate images in a folder
Generates COCO format annotations for easy review/correction
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from tqdm import tqdm
import shutil
from datetime import datetime

class AutoAnnotator:
    def __init__(self, model_path, confidence_threshold=0.25, iou_threshold=0.45):
        """
        Initialize auto-annotator
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: NMS IoU threshold
        """
        print(f"🔧 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Get class names from model
        if hasattr(self.model, 'names') and self.model.names:
            self.class_names = self.model.names
            self.class_id_to_name = {i: name for i, name in self.class_names.items()}
            self.class_name_to_id = {name: i for i, name in self.class_names.items()}
        else:
            # Default based on your dataset
            self.class_names = {
                0: 'enemy',
                1: 'money', 
                2: 'health',
                3: 'brotato',
                4: 'tree',
                5: 'chest'
            }
            self.class_id_to_name = self.class_names
            self.class_name_to_id = {v: k for k, v in self.class_names.items()}
        
        print(f"✅ Model loaded with {len(self.class_names)} classes:")
        for class_id, class_name in self.class_names.items():
            print(f"   Class {class_id}: {class_name}")
    
    def detect_and_annotate_image(self, image_path, save_visualization=True, 
                                 output_dir=None, min_conf_for_save=0.3):
        """
        Detect objects in a single image and return annotations
        
        Returns:
            dict with 'detections', 'image_info', and 'visualization_path'
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        height, width = image.shape[:2]
        filename = os.path.basename(image_path)
        
        # Run detection
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Extract detections
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.conf > min_conf_for_save:  # Only keep high-confidence detections
                    x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Convert to COCO format: [x_min, y_min, width, height]
                    x_min = x1
                    y_min = y1
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Ensure bbox is within image bounds
                    x_min = max(0, min(x_min, width-1))
                    y_min = max(0, min(y_min, height-1))
                    bbox_width = min(bbox_width, width - x_min)
                    bbox_height = min(bbox_height, height - y_min)
                    
                    # Skip invalid boxes
                    if bbox_width <= 0 or bbox_height <= 0:
                        continue
                    
                    detections.append({
                        'bbox': [x_min, y_min, bbox_width, bbox_height],
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': self.class_id_to_name.get(class_id, f'class_{class_id}'),
                        'area': bbox_width * bbox_height
                    })
        
        # Create visualization if requested
        vis_path = None
        if save_visualization and output_dir:
            vis_image = self.create_visualization(image.copy(), detections, filename)
            vis_path = os.path.join(output_dir, 'visualizations', filename)
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            cv2.imwrite(vis_path, vis_image)
        
        return {
            'detections': detections,
            'image_info': {
                'file_name': filename,
                'width': width,
                'height': height,
                'path': image_path
            },
            'visualization_path': vis_path
        }
    
    def create_visualization(self, image, detections, filename):
        """Create visualization image with detections"""
        colors = [
            (0, 0, 255),    # Red - enemy
            (0, 255, 255),  # Yellow - money
            (0, 255, 0),    # Green - health
            (255, 0, 0),    # Blue - brotato
            (128, 0, 128),  # Purple - tree
            (255, 165, 0)   # Orange - chest
        ]
        
        # Draw each detection
        for i, det in enumerate(detections):
            x_min, y_min, width, height = map(int, det['bbox'])
            x_max = x_min + width
            y_max = y_min + height
            
            # Get color
            class_id = det['class_id']
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(image, 
                         (x_min, y_min - label_size[1] - 10),
                         (x_min + label_size[0] + 10, y_min),
                         color, -1)
            
            # Text
            cv2.putText(image, label, 
                       (x_min + 5, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add watermark
        cv2.putText(image, "AUTO-ANNOTATED (REVIEW REQUIRED)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"Detections: {len(detections)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    
    def annotate_folder(self, input_folder, output_dir=None, 
                       formats=['coco', 'yolo', 'voc'], 
                       copy_images=False, min_conf_for_save=0.3):
        """
        Auto-annotate all images in a folder
        
        Args:
            input_folder: Folder containing images to annotate
            output_dir: Directory to save annotations
            formats: List of formats to export ('coco', 'yolo', 'voc')
            copy_images: Copy images to output directory
            min_conf_for_save: Minimum confidence to save a detection
        """
        print(f"\n📁 Processing folder: {input_folder}")
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_folder).glob(f'*{ext}'))
            image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No images found in {input_folder}")
            return
        
        print(f"📊 Found {len(image_files)} images")
        
        # Create output directory structure
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"auto_annotations_{timestamp}"
        
        output_dirs = {
            'root': output_dir,
            'images': os.path.join(output_dir, 'images'),
            'visualizations': os.path.join(output_dir, 'visualizations'),
            'annotations': os.path.join(output_dir, 'annotations')
        }
        
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"💾 Output will be saved to: {output_dir}")
        
        # Process each image
        all_results = []
        stats = {
            'total_images': len(image_files),
            'images_with_detections': 0,
            'total_detections': 0,
            'detections_by_class': {class_id: 0 for class_id in self.class_names.keys()}
        }
        
        print("\n🔍 Starting auto-annotation...")
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.detect_and_annotate_image(
                str(img_path), 
                save_visualization=True,
                output_dir=output_dir,
                min_conf_for_save=min_conf_for_save
            )
            
            if result:
                all_results.append(result)
                
                # Update statistics
                detections = result['detections']
                if detections:
                    stats['images_with_detections'] += 1
                    stats['total_detections'] += len(detections)
                    for det in detections:
                        class_id = det['class_id']
                        stats['detections_by_class'][class_id] = stats['detections_by_class'].get(class_id, 0) + 1
                
                # Copy image if requested
                if copy_images:
                    dest_path = os.path.join(output_dirs['images'], result['image_info']['file_name'])
                    shutil.copy2(str(img_path), dest_path)
        
        # Export annotations in requested formats
        print(f"\n📤 Exporting annotations...")
        
        for format_name in formats:
            if format_name == 'coco':
                self.export_coco_format(all_results, output_dirs['annotations'])
            elif format_name == 'yolo':
                self.export_yolo_format(all_results, output_dirs['annotations'])
            elif format_name == 'voc':
                self.export_voc_format(all_results, output_dirs['annotations'])
        
        # Print statistics
        self.print_statistics(stats, all_results)
        
        # Create review guide
        self.create_review_guide(output_dir, all_results)
        
        print(f"\n✅ Auto-annotation complete!")
        print(f"📁 Output directory: {os.path.abspath(output_dir)}")
        
        return output_dir, all_results
    
    def export_coco_format(self, results, output_dir):
        """Export annotations in COCO format"""
        print("  📝 Exporting COCO format...")
        
        # Create COCO structure
        coco_data = {
            'info': {
                'description': 'Auto-annotated dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().isoformat()
            },
            'licenses': [{'id': 0, 'name': 'Unknown', 'url': ''}],
            'categories': [],
            'images': [],
            'annotations': []
        }
        
        # Add categories
        for class_id, class_name in self.class_names.items():
            coco_data['categories'].append({
                'id': class_id,
                'name': class_name,
                'supercategory': ''
            })
        
        # Add images and annotations
        annotation_id = 0
        for img_idx, result in enumerate(results):
            img_info = result['image_info']
            
            # Add image
            coco_data['images'].append({
                'id': img_idx,
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'license': 0,
                'date_captured': ''
            })
            
            # Add annotations
            for det in result['detections']:
                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': img_idx,
                    'category_id': det['class_id'],
                    'bbox': det['bbox'],
                    'area': det['area'],
                    'segmentation': [],  # Empty for bounding boxes
                    'iscrowd': 0,
                    'confidence': det['confidence']  # Custom field
                })
                annotation_id += 1
        
        # Save to file
        output_path = os.path.join(output_dir, 'annotations_coco.json')
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"    ✅ Saved: {output_path}")
        return output_path
    
    def export_yolo_format(self, results, output_dir):
        """Export annotations in YOLO format"""
        print("  📝 Exporting YOLO format...")
        
        yolo_dir = os.path.join(output_dir, 'yolo_format')
        os.makedirs(yolo_dir, exist_ok=True)
        
        for result in results:
            img_info = result['image_info']
            filename = img_info['file_name']
            
            # Create label file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(yolo_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for det in result['detections']:
                    x_min, y_min, width, height = det['bbox']
                    img_width = img_info['width']
                    img_height = img_info['height']
                    
                    # Convert to YOLO format
                    x_center = (x_min + width / 2) / img_width
                    y_center = (y_min + height / 2) / img_height
                    width_norm = width / img_width
                    height_norm = height / img_height
                    
                    # Write to file
                    f.write(f"{det['class_id']} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        # Create dataset.yaml
        yaml_path = os.path.join(yolo_dir, 'dataset.yaml')
        yaml_content = f"""# Auto-annotated dataset
path: {os.path.abspath(yolo_dir)}
train: .  # All images in this directory
val: .    # Same directory for simplicity

# Classes
nc: {len(self.class_names)}
names: {self.class_names}
"""
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"    ✅ Saved YOLO format to: {yaml_path}")
        return yolo_dir
    
    def export_voc_format(self, results, output_dir):
        """Export annotations in Pascal VOC format (XML)"""
        print("  📝 Exporting VOC format...")
        
        voc_dir = os.path.join(output_dir, 'voc_format')
        os.makedirs(voc_dir, exist_ok=True)
        
        for result in results:
            img_info = result['image_info']
            filename = img_info['file_name']
            
            # Create XML file
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(voc_dir, xml_filename)
            
            # Create XML content
            xml_content = f"""<annotation>
    <folder>{voc_dir}</folder>
    <filename>{filename}</filename>
    <path>{img_info['path']}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{img_info['width']}</width>
        <height>{img_info['height']}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
"""
            
            for det in result['detections']:
                x_min, y_min, width, height = map(int, det['bbox'])
                x_max = x_min + width
                y_max = y_min + height
                
                xml_content += f"""    <object>
        <name>{det['class_name']}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <confidence>{det['confidence']:.4f}</confidence>
        <bndbox>
            <xmin>{x_min}</xmin>
            <ymin>{y_min}</ymin>
            <xmax>{x_max}</xmax>
            <ymax>{y_max}</ymax>
        </bndbox>
    </object>
"""
            
            xml_content += "</annotation>"
            
            with open(xml_path, 'w') as f:
                f.write(xml_content)
        
        print(f"    ✅ Saved VOC format to: {voc_dir}")
        return voc_dir
    
    def print_statistics(self, stats, results):
        """Print annotation statistics"""
        print(f"\n📊 AUTO-ANNOTATION STATISTICS")
        print("=" * 50)
        print(f"Total images processed: {stats['total_images']}")
        print(f"Images with detections: {stats['images_with_detections']} ({stats['images_with_detections']/stats['total_images']*100:.1f}%)")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}")
        
        print(f"\n📈 Detections by class:")
        for class_id, count in stats['detections_by_class'].items():
            if count > 0:
                class_name = self.class_id_to_name.get(class_id, f'class_{class_id}')
                percentage = count / stats['total_detections'] * 100 if stats['total_detections'] > 0 else 0
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Confidence distribution
        confidences = []
        for result in results:
            for det in result['detections']:
                confidences.append(det['confidence'])
        
        if confidences:
            print(f"\n🎯 Confidence statistics:")
            print(f"  Average confidence: {np.mean(confidences):.3f}")
            print(f"  Min confidence: {np.min(confidences):.3f}")
            print(f"  Max confidence: {np.max(confidences):.3f}")
            print(f"  Std confidence: {np.std(confidences):.3f}")
            
            # Histogram
            hist, bins = np.histogram(confidences, bins=10, range=(0, 1))
            print(f"\n  Confidence distribution:")
            for i in range(len(hist)):
                print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} detections")
    
    def create_review_guide(self, output_dir, results):
        """Create a review guide for manual verification - no emojis version"""
        print("\nCreating review guide...")
        
        guide_path = os.path.join(output_dir, 'REVIEW_GUIDE.md')
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("# Auto-Annotation Review Guide\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n")
            f.write(f"- Total images: {len(results)}\n")
            
            # Count images with/without detections
            images_with_detections = sum(1 for r in results if r['detections'])
            f.write(f"- Images with detections: {images_with_detections}\n")
            f.write(f"- Images without detections: {len(results) - images_with_detections}\n\n")
            
            f.write("## How to Review\n")
            f.write("1. Check visualizations in '/visualizations/' folder\n")
            f.write("2. Review high-confidence detections (confidence > 0.7) - usually correct\n")
            f.write("3. Review low-confidence detections (confidence < 0.3) - may need correction\n")
            f.write("4. Check for missed objects - compare with original images\n")
            f.write("5. Verify class labels - ensure correct class assignment\n\n")
            
            f.write("## Files to Review\n")
            f.write("| File | Detections | Notes |\n")
            f.write("|------|------------|-------|\n")
            
            for result in results[:50]:  # First 50 files
                filename = result['image_info']['file_name']
                det_count = len(result['detections'])
                vis_path = result.get('visualization_path', '')
                vis_rel = os.path.relpath(vis_path, output_dir) if vis_path else ''
                
                if det_count == 0:
                    f.write(f"| [{filename}]({vis_rel}) | 0 | No detections - check if empty |\n")
                elif det_count > 10:
                    f.write(f"| [{filename}]({vis_rel}) | {det_count} | Many detections - verify all |\n")
                else:
                    f.write(f"| [{filename}]({vis_rel}) | {det_count} | Review |\n")
            
            if len(results) > 50:
                f.write(f"| ... | ... | ... {len(results) - 50} more files ... |\n")
            
            f.write("\n## Correction Tools\n")
            f.write("- Use LabelImg for Pascal VOC format\n")
            f.write("- Use CVAT for online annotation\n")
            f.write("- Use Roboflow for collaborative annotation\n")
            f.write("- Use makesense.ai for free online tool\n\n")
            
            f.write("## Next Steps\n")
            f.write("1. Review all visualizations\n")
            f.write("2. Correct annotations in preferred tool\n")
            f.write("3. Retrain model with corrected annotations\n")
            f.write("4. Repeat process for continuous improvement\n")
        
        print(f"    Review guide saved: {guide_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Auto-annotate images using trained YOLO model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-annotate all images in a folder (default settings)
  python auto_annotate.py --model runs/detect/yolo_training/weights/best.pt --input unlabeled_images/
  
  # Auto-annotate with higher confidence threshold
  python auto_annotate.py --model best.pt --input images/ --conf 0.5
  
  # Export multiple formats
  python auto_annotate.py --model best.pt --input images/ --formats coco yolo
  
  # Copy images to output directory
  python auto_annotate.py --model best.pt --input images/ --copy-images
  
  # Set custom output directory
  python auto_annotate.py --model best.pt --input images/ --output my_annotations/
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Folder containing images to annotate')
    parser.add_argument('--output', type=str,
                       help='Output directory (default: auto_annotations_YYYYMMDD_HHMMSS)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--min-conf-save', type=float, default=0.3,
                       help='Minimum confidence to save a detection (default: 0.3)')
    parser.add_argument('--formats', nargs='+', default=['coco', 'yolo'],
                       choices=['coco', 'yolo', 'voc'],
                       help='Annotation formats to export (default: coco yolo)')
    parser.add_argument('--copy-images', action='store_true',
                       help='Copy images to output directory')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Do not create visualization images')
    
    args = parser.parse_args()
    
    # Check inputs
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"❌ Input folder not found: {args.input}")
        return
    
    print(f"🚀 Starting auto-annotation...")
    print(f"   Model: {args.model}")
    print(f"   Input: {args.input}")
    print(f"   Confidence threshold: {args.conf}")
    print(f"   Min confidence to save: {args.min_conf_save}")
    print(f"   Formats: {args.formats}")
    
    # Create annotator
    annotator = AutoAnnotator(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run annotation
    output_dir, results = annotator.annotate_folder(
        input_folder=args.input,
        output_dir=args.output,
        formats=args.formats,
        copy_images=args.copy_images,
        min_conf_for_save=args.min_conf_save
    )
    
    print(f"\n🎉 Auto-annotation complete!")
    print(f"\n📋 Next steps:")
    print(f"1. Review visualizations in: {os.path.join(output_dir, 'visualizations')}")
    print(f"2. Check review guide: {os.path.join(output_dir, 'REVIEW_GUIDE.md')}")
    print(f"3. Use annotations in: {os.path.join(output_dir, 'annotations')}")
    print(f"\n💡 Tip: Start by reviewing high-confidence detections first!")

if __name__ == '__main__':
    main()
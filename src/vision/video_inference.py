import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import torch

class VideoDetector:
    def __init__(self, model_path, class_names=None, confidence_threshold=0.25, iou_threshold=0.45, device=None):
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if class_names is None:
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            else:
                self.class_names = {
                    0: 'enemy',
                    1: 'money', 
                    2: 'health',
                    3: 'brotato',
                    4: 'tree',
                    5: 'chest'
                }
        else:
            self.class_names = class_names
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device.lower()
            if device == 'cuda' and not torch.cuda.is_available():
                print("CUDA requested but not available. Falling back to CPU.")
                device = 'cpu'

        self.device = device

        try:
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
        except Exception:
            pass

        gpu_info = ''
        if self.device.startswith('cuda') and torch.cuda.is_available():
            try:
                gpu_info = f" ({torch.cuda.get_device_name(0)})"
            except Exception:
                gpu_info = ''

        print(f"Model loaded successfully on {self.device}{gpu_info}")
        print(f"   Classes: {self.class_names}")
        print(f"   Confidence threshold: {self.conf_threshold}")
        print(f"   IoU threshold: {self.iou_threshold}")
    
    def draw_detections(self, frame, detections, show_labels=True, show_conf=True):
        annotated_frame = frame.copy()
        
        if detections and detections[0].boxes is not None:
            boxes = detections[0].boxes
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy_boxes = boxes.xyxy.cpu().numpy()
            
            colors = [
                (0, 0, 255),
                (0, 255, 255),
                (0, 255, 0),
                (255, 0, 0),
                (128, 0, 128),
                (255, 165, 0)
            ]
            
            for i, (box, conf, cls_id) in enumerate(zip(xyxy_boxes, confidences, class_ids)):
                if conf < self.conf_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                color_idx = cls_id % len(colors)
                color = colors[color_idx]
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label = ""
                if show_labels:
                    class_name = self.class_names.get(cls_id, f"Class {cls_id}")
                    label = class_name
                
                if show_conf:
                    if label:
                        label += f" {conf:.2f}"
                    else:
                        label = f"{conf:.2f}"
                
                if label:
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    label_y = max(y1 - 10, 20)
                    cv2.rectangle(annotated_frame, 
                                 (x1, label_y - label_size[1] - 5),
                                 (x1 + label_size[0] + 10, label_y + 5),
                                 color, -1)
                    
                    cv2.putText(annotated_frame, label, 
                               (x1 + 5, label_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video(self, video_path, output_path=None, 
                      show_live=True, save_output=True, 
                      frame_skip=0, max_frames=None):
        print(f"\nProcessing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Info:")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {total_frames}")
        
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_name = Path(video_path).stem
                output_path = f"outputs/{video_name}_detected_{timestamp}.mp4"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
            progress_bar = tqdm(total=total_frames, desc=f"Exporting {os.path.basename(output_path)}", unit="frame")
        
        frame_count = 0
        processed_count = 0
        detection_count = 0
        start_time = time.time()
        
        print(f"\nStarting video processing...")
        print("Press 'q' to quit, 'p' to pause, 's' to skip 10 frames")
        
        paused = False
        frame_skip_counter = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video reached")
                    break
                
                frame_count += 1
                frame_skip_counter += 1
                
                if frame_skip > 0 and frame_skip_counter <= frame_skip:
                    if save_output:
                        out.write(frame)
                        if 'progress_bar' in locals() and progress_bar is not None:
                            progress_bar.update(1)
                    continue
                frame_skip_counter = 0
                
                if max_frames is None or processed_count < max_frames:
                    results = self.model.predict(
                        source=frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False,
                        device=self.device
                    )
                    
                    processed_count += 1
                    
                    if results and results[0].boxes is not None:
                        detection_count += len(results[0].boxes)
                    
                    annotated_frame = self.draw_detections(frame, results)
                else:
                    annotated_frame = frame
                
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Detections: {detection_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    current_fps = processed_count / elapsed_time
                    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if save_output:
                    out.write(annotated_frame)
                    if 'progress_bar' in locals() and progress_bar is not None:
                        progress_bar.update(1)
                
                if show_live:
                    cv2.imshow('YOLO Video Detection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nProcessing stopped by user")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):
                for _ in range(10):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                print(f"Skipped to frame {frame_count}")
            elif key == ord('c'):
                screenshot_path = f"screenshots/frame_{frame_count:06d}.png"
                os.makedirs('screenshots', exist_ok=True)
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        cap.release()
        if save_output:
            out.release()
            if 'progress_bar' in locals() and progress_bar is not None:
                progress_bar.close()
            print(f"\nOutput video saved: {output_path}")
        
        cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        print(f"\nProcessing Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Processed frames: {processed_count}")
        print(f"  Total detections: {detection_count}")
        print(f"  Processing time: {elapsed_time:.2f} seconds")
        print(f"  Average FPS: {processed_count/elapsed_time:.2f}" if elapsed_time > 0 else "")
        print(f"  Detections per frame: {detection_count/processed_count:.2f}" if processed_count > 0 else "")
    
def main():
    parser = argparse.ArgumentParser(
        description='Run YOLO object detection on video files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_inference.py --model runs/detect/yolo_training/weights/best.pt --video input.mp4
  
  python video_inference.py --model best.pt --video input.mp4 --conf 0.5 --iou 0.3
  
  python video_inference.py --model best.pt --video input.mp4 --no-show
  
  python video_inference.py --model best.pt --video input.mp4 --max-frames 100
  
  python video_inference.py --model best.pt --video input.mp4 --frame-skip 1
  
  python video_inference.py --model best.pt --video input.mp4 --output my_output.mp4
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video file')
    input_group.add_argument('--image', type=str, help='Path to input image file')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help='Device to run on (cpu or cuda). If omitted, uses CUDA if available.')
    
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show live preview (faster processing)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output file')
    
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--frame-skip', type=int, default=0,
                       help='Process every N-th frame (0=all, 1=skip 1 frame, etc.)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum number of frames to process')
    
    parser.add_argument('--no-labels', action='store_true',
                       help='Do not show labels on bounding boxes')
    parser.add_argument('--no-conf', action='store_true',
                       help='Do not show confidence scores')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    detector = VideoDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    if hasattr(args, 'device') and args.device:
        detector = VideoDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou=args.iou,
            device=args.device
        )
    
    if args.video:
        detector.process_video(
            video_path=args.video,
            output_path=args.output,
            show_live=not args.no_show,
            save_output=not args.no_save,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames
        )
    elif args.image:
        print(f"Processing image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Cannot read image {args.image}")
            return
        
        results = detector.model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            device=detector.device
        )
        
        annotated_image = detector.draw_detections(
            image, results, 
            show_labels=not args.no_labels, 
            show_conf=not args.no_conf
        )
        
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = Path(args.image).stem
            output_path = f"outputs/{image_name}_detected_{timestamp}.jpg"
        
        os.makedirs('outputs', exist_ok=True)
        cv2.imwrite(output_path, annotated_image)
        print(f"Output saved: {output_path}")
        
        if not args.no_show:
            cv2.imshow('YOLO Image Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

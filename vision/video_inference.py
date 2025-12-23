"""
Run YOLO model inference on video files and render bounding boxes
"""

import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO
from pathlib import Path
import time
from datetime import datetime

class VideoDetector:
    def __init__(self, model_path, class_names=None, confidence_threshold=0.25, iou_threshold=0.45):
        """
        Initialize YOLO video detector
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            class_names: Dictionary mapping class IDs to names
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
        """
        print(f"🔧 Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Get class names from model if not provided
        if class_names is None:
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            else:
                # Default class names based on your dataset
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
        
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {self.class_names}")
        print(f"   Confidence threshold: {self.conf_threshold}")
        print(f"   IoU threshold: {self.iou_threshold}")
    
    def draw_detections(self, frame, detections, show_labels=True, show_conf=True):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input image/frame
            detections: YOLO detection results
            show_labels: Whether to show class labels
            show_conf: Whether to show confidence scores
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        if detections and detections[0].boxes is not None:
            boxes = detections[0].boxes
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy_boxes = boxes.xyxy.cpu().numpy()
            
            # Colors for different classes (BGR format)
            colors = [
                (0, 0, 255),    # Red - enemy
                (0, 255, 255),  # Yellow - money
                (0, 255, 0),    # Green - health
                (255, 0, 0),    # Blue - brotato
                (128, 0, 128),  # Purple - tree
                (255, 165, 0)   # Orange - chest
            ]
            
            for i, (box, conf, cls_id) in enumerate(zip(xyxy_boxes, confidences, class_ids)):
                if conf < self.conf_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Get color for this class
                color_idx = cls_id % len(colors)
                color = colors[color_idx]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text
                label = ""
                if show_labels:
                    class_name = self.class_names.get(cls_id, f"Class {cls_id}")
                    label = class_name
                
                if show_conf:
                    if label:
                        label += f" {conf:.2f}"
                    else:
                        label = f"{conf:.2f}"
                
                # Draw label background
                if label:
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    label_y = max(y1 - 10, 20)
                    cv2.rectangle(annotated_frame, 
                                 (x1, label_y - label_size[1] - 5),
                                 (x1 + label_size[0] + 10, label_y + 5),
                                 color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, 
                               (x1 + 5, label_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video(self, video_path, output_path=None, 
                      show_live=True, save_output=True, 
                      frame_skip=0, max_frames=None):
        """
        Process video file with YOLO model
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (if None, auto-generate)
            show_live: Show live preview during processing
            save_output: Save output video
            frame_skip: Process every N-th frame (0 = all frames)
            max_frames: Maximum number of frames to process
        """
        print(f"\n🎬 Processing video: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info:")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {total_frames}")
        
        # Setup video writer if saving output
        if save_output:
            if output_path is None:
                # Auto-generate output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_name = Path(video_path).stem
                output_path = f"outputs/{video_name}_detected_{timestamp}.mp4"
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
        # Statistics
        frame_count = 0
        processed_count = 0
        detection_count = 0
        start_time = time.time()
        
        print(f"\n▶️ Starting video processing...")
        print("Press 'q' to quit, 'p' to pause, 's' to skip 10 frames")
        
        paused = False
        frame_skip_counter = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ End of video reached")
                    break
                
                frame_count += 1
                frame_skip_counter += 1
                
                # Skip frames if specified
                if frame_skip > 0 and frame_skip_counter <= frame_skip:
                    if save_output:
                        out.write(frame)
                    continue
                frame_skip_counter = 0
                
                # Process frame
                if max_frames is None or processed_count < max_frames:
                    # Run YOLO inference
                    results = self.model.predict(
                        source=frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False,
                        device='cuda' if self.model.device.type == 'cuda' else 'cpu'
                    )
                    
                    processed_count += 1
                    
                    # Count detections
                    if results and results[0].boxes is not None:
                        detection_count += len(results[0].boxes)
                    
                    # Draw detections
                    annotated_frame = self.draw_detections(frame, results)
                else:
                    annotated_frame = frame
                
                # Add frame counter and FPS
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Detections: {detection_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Calculate and display FPS
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    current_fps = processed_count / elapsed_time
                    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save output
                if save_output:
                    out.write(annotated_frame)
                
                # Show live preview
                if show_live:
                    cv2.imshow('YOLO Video Detection', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                print("\n⏹️ Processing stopped by user")
                break
            elif key == ord('p'):  # Pause/Resume
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
            elif key == ord('s'):  # Skip 10 frames
                for _ in range(10):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                print(f"⏭️ Skipped to frame {frame_count}")
            elif key == ord('c'):  # Capture screenshot
                screenshot_path = f"screenshots/frame_{frame_count:06d}.png"
                os.makedirs('screenshots', exist_ok=True)
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
        
        # Cleanup
        cap.release()
        if save_output:
            out.release()
            print(f"\n✅ Output video saved: {output_path}")
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        elapsed_time = time.time() - start_time
        print(f"\n📊 Processing Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Processed frames: {processed_count}")
        print(f"  Total detections: {detection_count}")
        print(f"  Processing time: {elapsed_time:.2f} seconds")
        print(f"  Average FPS: {processed_count/elapsed_time:.2f}" if elapsed_time > 0 else "")
        print(f"  Detections per frame: {detection_count/processed_count:.2f}" if processed_count > 0 else "")
    
    def process_webcam(self, camera_id=0, show_live=True, save_output=False):
        """
        Process live webcam feed with YOLO model
        
        Args:
            camera_id: Camera device ID (0 for default)
            show_live: Show live preview
            save_output: Save output to file
        """
        print(f"📹 Starting webcam detection (Camera ID: {camera_id})")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open camera {camera_id}")
            return
        
        # Get webcam properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Camera Info: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if saving output
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/webcam_{timestamp}.mp4"
            os.makedirs('outputs', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Recording to: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        print("\n▶️ Starting webcam feed...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame from camera")
                break
            
            frame_count += 1
            
            # Run YOLO inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device='cuda' if self.model.device.type == 'cuda' else 'cpu'
            )
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, results)
            
            # Add FPS counter
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save frame if recording
            if save_output:
                out.write(annotated_frame)
            
            # Show live preview
            if show_live:
                cv2.imshow('YOLO Webcam Detection', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️ Webcam feed stopped")
                break
            elif key == ord('s'):
                screenshot_path = f"screenshots/webcam_{frame_count:06d}.png"
                os.makedirs('screenshots', exist_ok=True)
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
        
        # Cleanup
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Webcam Statistics:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total time: {elapsed_time:.2f} seconds")
        print(f"  Average FPS: {fps:.2f}")

def main():
    parser = argparse.ArgumentParser(
        description='Run YOLO object detection on video files or webcam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file with default settings
  python video_inference.py --model runs/detect/yolo_training/weights/best.pt --video input.mp4
  
  # Process video with custom confidence threshold
  python video_inference.py --model best.pt --video input.mp4 --conf 0.5 --iou 0.3
  
  # Process webcam feed
  python video_inference.py --model best.pt --webcam
  
  # Process video without live preview (faster)
  python video_inference.py --model best.pt --video input.mp4 --no-show
  
  # Process only first 100 frames
  python video_inference.py --model best.pt --video input.mp4 --max-frames 100
  
  # Skip every other frame (process at 50% speed)
  python video_inference.py --model best.pt --video input.mp4 --frame-skip 1
  
  # Process video and save output with specific name
  python video_inference.py --model best.pt --video input.mp4 --output my_output.mp4
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video file')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam feed')
    input_group.add_argument('--image', type=str, help='Path to input image file')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    
    # Output arguments
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show live preview (faster processing)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output file')
    
    # Processing arguments
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--frame-skip', type=int, default=0,
                       help='Process every N-th frame (0=all, 1=skip 1 frame, etc.)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum number of frames to process')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera ID for webcam (default: 0)')
    
    # Display arguments
    parser.add_argument('--no-labels', action='store_true',
                       help='Do not show labels on bounding boxes')
    parser.add_argument('--no-conf', action='store_true',
                       help='Do not show confidence scores')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    # Create detector
    detector = VideoDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Process based on input type
    if args.video:
        detector.process_video(
            video_path=args.video,
            output_path=args.output,
            show_live=not args.no_show,
            save_output=not args.no_save,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames
        )
    elif args.webcam:
        detector.process_webcam(
            camera_id=args.camera_id,
            show_live=not args.no_show,
            save_output=not args.no_save
        )
    elif args.image:
        print(f"🖼️ Processing image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"❌ Error: Cannot read image {args.image}")
            return
        
        # Run detection
        results = detector.model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )
        
        # Draw detections
        annotated_image = detector.draw_detections(
            image, results, 
            show_labels=not args.no_labels, 
            show_conf=not args.no_conf
        )
        
        # Save output
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = Path(args.image).stem
            output_path = f"outputs/{image_name}_detected_{timestamp}.jpg"
        
        os.makedirs('outputs', exist_ok=True)
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Output saved: {output_path}")
        
        # Show image
        if not args.no_show:
            cv2.imshow('YOLO Image Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
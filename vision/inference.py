from ultralytics import YOLO
import cv2
import os

def run_inference(model_path, image_path, output_dir='inference_results'):
    """
    Run inference on single image or directory
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device='cuda',
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name='predictions'
    )
    
    # Display results
    for result in results:
        if result.boxes is not None:
            print(f"\n📊 Detection results for: {result.path}")
            print(f"Detected {len(result.boxes)} objects")
            
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                print(f"  Object {i+1}: Class={cls}, Confidence={conf:.2f}, BBox={xyxy}")
    
    # Optionally show image with detections
    if isinstance(image_path, str) and os.path.isfile(image_path):
        result_img = results[0].plot()
        cv2.imshow('Detection Results', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results

if __name__ == '__main__':
    # Path to your trained model
    model_path = 'runs/detect/yolo_training/weights/best.pt'
    
    # Path to image or directory for inference
    input_path = 'test_images/'  # or 'test_image.png'
    
    if os.path.exists(model_path):
        run_inference(model_path, input_path)
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first or specify correct path.")
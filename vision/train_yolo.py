from ultralytics import YOLO
import os
import yaml

def train_yolo_model():
    """
    Train YOLOv8 model on custom dataset
    """
    
    # Load dataset configuration
    with open('yolo_dataset/dataset.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print("📊 Dataset Configuration:")
    print(f"  Path: {dataset_config['path']}")
    print(f"  Train: {dataset_config['train']}")
    print(f"  Val: {dataset_config['val']}")
    print(f"  Classes: {dataset_config['names']}")
    
    # Choose YOLO model (options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
    model_name = "yolov8s.pt"  # Small model for faster training
    
    print(f"\n🚀 Starting training with {model_name}...")
    
    # Load a pretrained model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data='yolo_dataset/dataset.yaml',  # dataset config
        epochs=100,                         # number of epochs
        imgsz=640,                         # image size
        batch=16,                          # batch size
        workers=4,                         # number of workers
        device='cuda',                     # use GPU if available
        project='runs/detect',             # save results to project
        name='yolo_training',              # experiment name
        exist_ok=True,                     # overwrite existing experiment
        verbose=True,                      # print progress
        # Additional parameters (optional):
        lr0=0.01,                          # initial learning rate
        lrf=0.01,                          # final learning rate
        momentum=0.937,                    # SGD momentum
        weight_decay=0.0005,               # optimizer weight decay
        warmup_epochs=3,                   # warmup epochs
        warmup_momentum=0.8,               # warmup momentum
        warmup_bias_lr=0.1,                # warmup bias learning rate
        box=7.5,                           # box loss gain
        cls=0.5,                           # cls loss gain
        dfl=1.5,                           # dfl loss gain
        hsv_h=0.015,                       # image HSV-Hue augmentation
        hsv_s=0.7,                         # image HSV-Saturation augmentation
        hsv_v=0.4,                         # image HSV-Value augmentation
        degrees=0.0,                       # image rotation (+/- deg)
        translate=0.1,                     # image translation (+/- fraction)
        scale=0.5,                         # image scale (+/- gain)
        shear=0.0,                         # image shear (+/- deg)
        perspective=0.0,                   # image perspective (+/- fraction)
        flipud=0.0,                        # image flip up-down (probability)
        fliplr=0.5,                        # image flip left-right (probability)
        mosaic=1.0,                        # image mosaic (probability)
        mixup=0.0,                         # image mixup (probability)
        copy_paste=0.0,                    # segment copy-paste (probability)
    )
    
    print("✅ Training completed!")
    return results

def evaluate_model(model_path='runs/detect/yolo_training/weights/best.pt'):
    """
    Evaluate trained model
    """
    print("\n📊 Evaluating model...")
    
    model = YOLO(model_path)
    
    # Evaluate on validation set
    metrics = model.val(
        data='yolo_dataset/dataset.yaml',
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        device='cuda',
        plots=True,  # generate plots
        save_json=True,  # save results to JSON
    )
    
    return metrics

def predict_on_sample(model_path='runs/detect/yolo_training/weights/best.pt'):
    """
    Run inference on sample images
    """
    print("\n🎯 Running inference on sample images...")
    
    model = YOLO(model_path)
    
    # Get sample images from validation set
    sample_images = []
    val_dir = 'yolo_dataset/images/val'
    if os.path.exists(val_dir):
        sample_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir)[:3]]
    
    if sample_images:
        # Run prediction
        results = model.predict(
            source=sample_images,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            device='cuda',
            save=True,
            save_txt=True,
            save_conf=True,
            show_labels=True,
            show_conf=True,
            line_width=2,
        )
        
        # Display results
        for i, result in enumerate(results):
            print(f"\nImage {i+1}: {result.path}")
            print(f"  Detected {len(result.boxes)} objects")
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    print(f"    Class {cls}: confidence={conf:.2f}")
    
    return results

if __name__ == '__main__':
    # Train the model
    train_results = train_yolo_model()
    
    # Evaluate the model
    eval_results = evaluate_model()
    
    # Run inference
    predict_results = predict_on_sample()
    
    print("\n✅ All tasks completed!")
    print(f"Model saved in: runs/detect/yolo_training/")
    print(f"Best weights: runs/detect/yolo_training/weights/best.pt")
import threading
import numpy as np
from pynput import keyboard
from ultralytics import YOLO
import torch
import time
import cv2

from overlay.visualize import render_boxes, format_results
from overlay.overlay_detection import capture_overlay, press_key, select_game_window

class ImageDetector:
    def __init__(self, model_path, class_names=None, confidence_threshold=0.25, iou_threshold=0.45, device='cuda'):
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if class_names is None:
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
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
        print(f"Classes: {self.class_names}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")

if __name__ == "__main__":
    detector = ImageDetector(
        model_path=r'vision\runs\detect\yolo_training\weights\last.pt',
        confidence_threshold=0.25,
        iou_threshold=0.45,
        device='cuda'
    )

    game_window = select_game_window("Brotato")

    if game_window:
        left, top, width, height = (
            game_window.left,
            game_window.top,
            game_window.width,
            game_window.height
        )
        
        stop_pressing = threading.Event()
        def on_press(key):
            try:
                if key.char == 'q':
                    stop_pressing.set()
            except AttributeError:
                pass
        
        def press_keys_randomly():
            listener = keyboard.Listener(on_press=on_press)
            listener.start()            
            while not stop_pressing.is_set():
                overlay_image = capture_overlay(game_window=game_window)
                overlay_np = cv2.cvtColor(np.array(overlay_image), cv2.COLOR_RGB2BGR)
                results = detector.model.predict(
                    source=overlay_np,
                    conf=detector.conf_threshold,
                    iou=detector.iou_threshold,
                    device=detector.device,
                    verbose=False
                )

                print(format_results(results, class_names=detector.class_names, conf_threshold=detector.conf_threshold))

                # try:
                #     render_boxes(overlay_np, results=results, class_names=detector.class_names, window_name='Detections', wait=1)
                # except Exception as e:
                #     print(f"Error rendering boxes: {e}")
            
            listener.stop()
        
        key_thread = threading.Thread(target=press_keys_randomly, daemon=True)
        key_thread.start()
        key_thread.join()

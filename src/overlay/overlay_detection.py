import pygetwindow as gw
import time
import random
import threading
from pynput import keyboard
from pynput.keyboard import Key, Controller
import mss
import numpy as np
from PIL import Image
import cv2

key_controller = Controller()

def press_key(key_name, seconds_holding):
    """Press and hold a key for specified duration"""
    special_keys = {
        'esc': Key.esc,
        'tab': Key.tab,
        'shift': Key.shift,
        'ctrl': Key.ctrl,
        'alt': Key.alt,
        'enter': Key.enter,
        'space': Key.space,
    }
    
    key = special_keys.get(key_name, key_name)
    key_controller.press(key)
    time.sleep(seconds_holding)
    key_controller.release(key)

def select_game_window(window_title="Your Game Name"):
    """Select game window by title"""
    windows = gw.getWindowsWithTitle(window_title)
    
    if not windows:
        print(f"No window found with title: {window_title}")
        print("Available windows:")
        for win in gw.getAllWindows():
            if win.title:
                print(f"  - {win.title}")
        return None
    
    game_window = windows[0]
    
    if game_window.isMinimized:
        game_window.restore()
    
    game_window.activate()
    time.sleep(0.5)
    
    return game_window

def capture_overlay(game_window=None, region=None):
    """
    Capture an overlay or region of the screen using MSS.
    
    Args:
        game_window: Optional pygetwindow window object to capture from
        region: Optional dict with 'left', 'top', 'width', 'height' keys
               If None and game_window is None, captures entire screen
    
    Returns:
        PIL Image object of the captured region
    """
    sct = mss.mss()

    # Build region from window or monitors and ensure integer values
    if region is None:
        if game_window:
            left = int(game_window.left)
            top = int(game_window.top)
            width = int(game_window.width)
            height = int(game_window.height)
        else:
            monitor = sct.monitors[1]
            left = int(monitor['left'])
            top = int(monitor['top'])
            width = int(monitor['width'])
            height = int(monitor['height'])

        region = {'left': left, 'top': top, 'width': width, 'height': height}

    # Sanitize region values
    region['left'] = int(region.get('left', 0))
    region['top'] = int(region.get('top', 0))
    region['width'] = max(1, int(region.get('width', 1)))
    region['height'] = max(1, int(region.get('height', 1)))

    # Try primary MSS capture, then fall back to PIL.ImageGrab, else return a blank image
    try:
        screenshot = sct.grab(region)
        image = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        return image
    except Exception as e:
        print(f"mss capture failed: {e}. Trying fallback capture methods.")
        try:
            # PIL's ImageGrab is sometimes more tolerant in different environments
            from PIL import ImageGrab
            bbox = (
                region['left'],
                region['top'],
                region['left'] + region['width'],
                region['top'] + region['height']
            )
            pil_img = ImageGrab.grab(bbox=bbox)
            return pil_img.convert('RGB')
        except Exception as e2:
            print(f"PIL fallback failed: {e2}. Returning blank image of requested size.")
            w = region['width']
            h = region['height']
            return Image.new('RGB', (w, h), (0, 0, 0))

if __name__ == "__main__":
    game_window = select_game_window("Brotato")
    if game_window:
        left, top, width, height = (
            game_window.left,
            game_window.top,
            game_window.width,
            game_window.height
        )
        
        keys_to_press = ['w', 'a', 's', 'd']
        stop_pressing = threading.Event()
        
        def on_press(key):
            try:
                if key.char == 'q':
                    print("Stopping key presses...")
                    stop_pressing.set()
            except AttributeError:
                pass
        
        def press_keys_randomly():
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
            
            press_key('esc', 0.1)
            time.sleep(0.5)
            
            cv2.namedWindow('Overlay Capture', cv2.WINDOW_NORMAL)
            
            print("Started pressing random keys. Press 'q' to stop.")
            while not stop_pressing.is_set():
                random_key = random.choice(keys_to_press)
                hold_time = random.uniform(1.0, 1.5)
                print(f"Pressing {random_key} for {hold_time:.2f}s...")
                #press_key(random_key, hold_time)
                time.sleep(random.uniform(0.1, 0.3))
                
                captured_image = capture_overlay(game_window=game_window)
                frame = cv2.cvtColor(np.array(captured_image), cv2.COLOR_RGB2BGR)
                
                cv2.imshow('Overlay Capture', frame)
                cv2.waitKey(1)
            
            listener.stop()
            print("Key pressing stopped.")
            cv2.destroyAllWindows()
        
        key_thread = threading.Thread(target=press_keys_randomly, daemon=True)
        key_thread.start()
        key_thread.join()
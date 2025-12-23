import pygetwindow as gw
import pyautogui
import time
import random
import threading
from pynput import keyboard
from pynput.keyboard import Key, Controller

key_controller = Controller()

def press_key(key_name, seconds_holding):
    """Press and hold a key for specified duration"""
    # Map special key names to pynput Key enum
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
    
    # Activate and bring to front
    game_window.activate()
    time.sleep(0.5)  # Wait for window to activate
    
    return game_window

# Usage
game_window = select_game_window("Brotato")
if game_window:
    # Now you can capture this specific window
    left, top, width, height = (
        game_window.left,
        game_window.top,
        game_window.width,
        game_window.height
    )
    
    # Start random key pressing
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
        
        # Press ESC once
        press_key('esc', 0.1)
        time.sleep(0.5)
        
        print("Started pressing random keys. Press 'q' to stop.")
        while not stop_pressing.is_set():
            random_key = random.choice(keys_to_press)
            hold_time = random.uniform(1.0, 1.5)
            print(f"Pressing {random_key} for {hold_time:.2f}s...")
            press_key(random_key, hold_time)
            time.sleep(random.uniform(0.1, 0.3))  # Delay before next key
        
        listener.stop()
        print("Key pressing stopped.")
    
    # Run in a separate thread
    key_thread = threading.Thread(target=press_keys_randomly, daemon=True)
    key_thread.start()
    key_thread.join()  # Wait for the thread to finish
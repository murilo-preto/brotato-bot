"""
Interactive tool to review and correct auto-generated annotations
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from PIL import Image, ImageTk
import shutil

class AnnotationReviewer:
    def __init__(self, annotations_dir):
        self.annotations_dir = annotations_dir
        self.visualizations_dir = os.path.join(annotations_dir, 'visualizations')
        self.images_dir = os.path.join(annotations_dir, 'images')
        
        # Load COCO annotations if available
        self.coco_path = os.path.join(annotations_dir, 'annotations', 'annotations_coco.json')
        self.annotations = self.load_annotations()
        
        # Get list of images
        self.image_files = self.get_image_files()
        self.current_index = 0
        
        # Review status
        self.reviewed = set()
        self.corrections = {}
        
        # Colors for classes
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Yellow
        ]
    
    def load_annotations(self):
        """Load annotations from COCO format"""
        if os.path.exists(self.coco_path):
            with open(self.coco_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_image_files(self):
        """Get list of image files to review"""
        image_files = []
        
        # Check visualizations first (these show detections)
        if os.path.exists(self.visualizations_dir):
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                image_files.extend(Path(self.visualizations_dir).glob(f'*{ext}'))
        
        # Fall back to original images
        if not image_files and os.path.exists(self.images_dir):
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                image_files.extend(Path(self.images_dir).glob(f'*{ext}'))
        
        return sorted(image_files)
    
    def get_image_annotations(self, image_filename):
        """Get annotations for a specific image"""
        if not self.annotations:
            return []
        
        # Find image ID
        image_id = None
        for img in self.annotations['images']:
            if img['file_name'] == image_filename:
                image_id = img['id']
                break
        
        if image_id is None:
            return []
        
        # Get annotations for this image
        img_annotations = []
        for ann in self.annotations['annotations']:
            if ann['image_id'] == image_id:
                img_annotations.append(ann)
        
        return img_annotations
    
    def mark_as_reviewed(self, image_path, status='approved', notes=''):
        """Mark an image as reviewed"""
        filename = os.path.basename(image_path)
        self.reviewed.add(filename)
        self.corrections[filename] = {
            'status': status,
            'notes': notes,
            'timestamp': str(np.datetime64('now'))
        }
    
    def save_progress(self):
        """Save review progress"""
        progress_path = os.path.join(self.annotations_dir, 'review_progress.json')
        progress = {
            'reviewed': list(self.reviewed),
            'corrections': self.corrections,
            'current_index': self.current_index
        }
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print(f"💾 Progress saved to: {progress_path}")
    
    def load_progress(self):
        """Load review progress"""
        progress_path = os.path.join(self.annotations_dir, 'review_progress.json')
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
            
            self.reviewed = set(progress.get('reviewed', []))
            self.corrections = progress.get('corrections', {})
            self.current_index = progress.get('current_index', 0)
            
            print(f"📖 Loaded progress: {len(self.reviewed)} images reviewed")
            return True
        
        return False

class ReviewGUI:
    def __init__(self, reviewer):
        self.reviewer = reviewer
        self.root = tk.Tk()
        self.root.title("Annotation Review Tool")
        self.root.geometry("1400x900")
        
        # Variables
        self.current_image = None
        self.current_photo = None
        
        # Setup GUI
        self.setup_gui()
        
        # Load first image
        self.load_current_image()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        
        # Information panel
        info_frame = ttk.LabelFrame(main_frame, text="Image Information", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=10, width=50)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        
        # Navigation buttons
        ttk.Button(controls_frame, text="← Previous", 
                  command=self.previous_image).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(controls_frame, text="Next →", 
                  command=self.next_image).grid(row=0, column=1, padx=5, pady=5)
        
        # Review buttons
        ttk.Button(controls_frame, text="✓ Approve", 
                  command=lambda: self.mark_reviewed('approved'),
                  style="Success.TButton").grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(controls_frame, text="✗ Reject", 
                  command=lambda: self.mark_reviewed('rejected'),
                  style="Danger.TButton").grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(controls_frame, text="⚠️ Needs Correction", 
                  command=lambda: self.mark_reviewed('needs_correction')).grid(row=1, column=2, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, 
                                           variable=self.progress_var,
                                           maximum=len(self.reviewer.image_files))
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Stats label
        self.stats_label = ttk.Label(controls_frame, text="")
        self.stats_label.grid(row=3, column=0, columnspan=3, pady=5)
        
        # Notes
        ttk.Label(controls_frame, text="Notes:").grid(row=4, column=0, sticky=tk.W, padx=5)
        self.notes_entry = ttk.Entry(controls_frame, width=40)
        self.notes_entry.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        
        # Save and quit buttons
        ttk.Button(controls_frame, text="💾 Save Progress", 
                  command=self.save_progress).grid(row=5, column=0, padx=5, pady=10)
        ttk.Button(controls_frame, text="📤 Export Corrected", 
                  command=self.export_corrected).grid(row=5, column=1, padx=5, pady=10)
        ttk.Button(controls_frame, text="❌ Quit", 
                  command=self.quit_app).grid(row=5, column=2, padx=5, pady=10)
        
        # Configure styles
        style = ttk.Style()
        style.configure("Success.TButton", foreground="green")
        style.configure("Danger.TButton", foreground="red")
        
        # Bind keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('a', lambda e: self.mark_reviewed('approved'))
        self.root.bind('r', lambda e: self.mark_reviewed('rejected'))
        self.root.bind('c', lambda e: self.mark_reviewed('needs_correction'))
        self.root.bind('<Control-s>', lambda e: self.save_progress())
        
        # Update progress
        self.update_progress()
    
    def load_current_image(self):
        """Load current image"""
        if not self.reviewer.image_files:
            messagebox.showinfo("No Images", "No images found to review.")
            return
        
        if self.reviewer.current_index >= len(self.reviewer.image_files):
            self.reviewer.current_index = 0
        
        image_path = self.reviewer.image_files[self.reviewer.current_index]
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            messagebox.showerror("Error", f"Could not load image: {image_path}")
            return
        
        # Resize for display
        height, width = img.shape[:2]
        max_size = 800
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Convert to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Convert to PhotoImage
        self.current_photo = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.current_photo)
        
        # Update info
        self.update_info(image_path)
        
        # Update progress
        self.update_progress()
    
    def update_info(self, image_path):
        """Update information panel"""
        filename = os.path.basename(image_path)
        annotations = self.reviewer.get_image_annotations(filename)
        
        info_text = f"Image: {filename}\n"
        info_text += f"Path: {image_path}\n"
        info_text += f"Detections: {len(annotations)}\n\n"
        
        if annotations:
            info_text += "Detected objects:\n"
            for i, ann in enumerate(annotations[:10]):  # Show first 10
                class_id = ann['category_id']
                confidence = ann.get('confidence', 'N/A')
                bbox = ann['bbox']
                info_text += f"  {i+1}. Class {class_id}, Conf: {confidence:.2f}, BBox: {bbox}\n"
            
            if len(annotations) > 10:
                info_text += f"  ... and {len(annotations) - 10} more\n"
        else:
            info_text += "No detections found\n"
        
        # Review status
        if filename in self.reviewer.reviewed:
            status = self.reviewer.corrections.get(filename, {}).get('status', 'reviewed')
            notes = self.reviewer.corrections.get(filename, {}).get('notes', '')
            info_text += f"\nStatus: {status.upper()}\n"
            if notes:
                info_text += f"Notes: {notes}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
    
    def update_progress(self):
        """Update progress bar and stats"""
        total = len(self.reviewer.image_files)
        reviewed = len(self.reviewer.reviewed)
        current = self.reviewer.current_index + 1
        
        # Update progress bar
        self.progress_var.set(reviewed)
        self.progress_bar.configure(maximum=total)
        
        # Update stats
        stats_text = f"Image {current}/{total} | Reviewed: {reviewed}/{total} ({reviewed/total*100:.1f}%)"
        self.stats_label.config(text=stats_text)
    
    def previous_image(self):
        """Go to previous image"""
        if self.reviewer.current_index > 0:
            self.reviewer.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.reviewer.current_index < len(self.reviewer.image_files) - 1:
            self.reviewer.current_index += 1
            self.load_current_image()
    
    def mark_reviewed(self, status):
        """Mark current image as reviewed"""
        if not self.reviewer.image_files:
            return
        
        image_path = self.reviewer.image_files[self.reviewer.current_index]
        filename = os.path.basename(image_path)
        notes = self.notes_entry.get()
        
        self.reviewer.mark_as_reviewed(image_path, status, notes)
        self.notes_entry.delete(0, tk.END)
        
        # Auto-advance to next image
        if self.reviewer.current_index < len(self.reviewer.image_files) - 1:
            self.reviewer.current_index += 1
            self.load_current_image()
        else:
            self.update_info(image_path)
            self.update_progress()
            messagebox.showinfo("Complete", "All images have been reviewed!")
    
    def save_progress(self):
        """Save review progress"""
        self.reviewer.save_progress()
        messagebox.showinfo("Saved", "Progress saved successfully!")
    
    def export_corrected(self):
        """Export corrected annotations"""
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select output directory")
        if not output_dir:
            return
        
        # Create corrected annotations
        corrected = {
            'approved': [],
            'needs_correction': [],
            'rejected': []
        }
        
        for filename, data in self.reviewer.corrections.items():
            status = data['status']
            if status in corrected:
                corrected[status].append(filename)
        
        # Save summary
        summary_path = os.path.join(output_dir, 'review_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(corrected, f, indent=2)
        
        messagebox.showinfo("Exported", f"Review summary exported to:\n{summary_path}")
    
    def quit_app(self):
        """Quit application"""
        if messagebox.askyesno("Quit", "Save progress before quitting?"):
            self.save_progress()
        self.root.quit()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Review auto-generated annotations')
    parser.add_argument('--annotations-dir', type=str, required=True,
                       help='Directory containing auto-annotations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotations_dir):
        print(f"❌ Annotations directory not found: {args.annotations_dir}")
        return
    
    # Create reviewer
    reviewer = AnnotationReviewer(args.annotations_dir)
    
    # Try to load progress
    reviewer.load_progress()
    
    # Start GUI
    app = ReviewGUI(reviewer)
    app.root.mainloop()

if __name__ == '__main__':
    main()
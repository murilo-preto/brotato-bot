import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def monitor_training(runs_dir='runs/detect'):
    """
    Monitor and visualize training progress
    """
    
    # Find all training runs
    runs = glob.glob(os.path.join(runs_dir, '*/'))
    
    if not runs:
        print("No training runs found!")
        return
    
    latest_run = max(runs, key=os.path.getmtime)
    print(f"📈 Monitoring latest run: {latest_run}")
    
    # Check for results CSV
    results_csv = os.path.join(latest_run, 'results.csv')
    
    if os.path.exists(results_csv):
        # Load results
        df = pd.read_csv(results_csv)
        
        # Plot metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)
        
        # Plot loss curves
        if 'train/box_loss' in df.columns:
            axes[0, 0].plot(df['train/box_loss'], label='Train Box Loss')
            axes[0, 0].plot(df['val/box_loss'], label='Val Box Loss')
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        if 'train/cls_loss' in df.columns:
            axes[0, 1].plot(df['train/cls_loss'], label='Train Class Loss')
            axes[0, 1].plot(df['val/cls_loss'], label='Val Class Loss')
            axes[0, 1].set_title('Class Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        if 'train/dfl_loss' in df.columns:
            axes[0, 2].plot(df['train/dfl_loss'], label='Train DFL Loss')
            axes[0, 2].plot(df['val/dfl_loss'], label='Val DFL Loss')
            axes[0, 2].set_title('DFL Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Plot metrics
        if 'metrics/precision(B)' in df.columns:
            axes[1, 0].plot(df['metrics/precision(B)'], label='Precision')
            axes[1, 0].set_title('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        if 'metrics/recall(B)' in df.columns:
            axes[1, 1].plot(df['metrics/recall(B)'], label='Recall')
            axes[1, 1].set_title('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 2].plot(df['metrics/mAP50(B)'], label='mAP50')
            axes[1, 2].set_title('mAP50')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
        
        # Print summary
        print("\n📊 Training Summary:")
        print(f"Total epochs: {len(df)}")
        
        if 'metrics/mAP50(B)' in df.columns:
            best_map = df['metrics/mAP50(B)'].max()
            print(f"Best mAP50: {best_map:.4f}")
        
        if 'metrics/precision(B)' in df.columns:
            best_precision = df['metrics/precision(B)'].max()
            print(f"Best precision: {best_precision:.4f}")
    
    # Show model files
    weights_dir = os.path.join(latest_run, 'weights')
    if os.path.exists(weights_dir):
        print(f"\n📁 Model files in {weights_dir}:")
        for file in os.listdir(weights_dir):
            file_path = os.path.join(weights_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    monitor_training()
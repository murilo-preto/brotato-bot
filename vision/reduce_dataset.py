import os
import sys
from pathlib import Path

def reduce_dataset(directory, keep_every=30, dry_run=False):
    """
    Reduce the dataset by keeping only 1 out of every 'keep_every' images.
    
    Args:
        directory: Path to the directory containing images
        keep_every: Keep 1 image every N images (default: 30)
        dry_run: If True, only show what would be deleted without actually deleting
    """
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    for file in Path(directory).iterdir():
        if file.suffix.lower() in image_extensions and file.is_file():
            image_files.append(file)
    
    # Sort files to maintain temporal order
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Calculate which files to keep
    files_to_keep = []
    files_to_delete = []
    
    for i, file in enumerate(image_files):
        if i % keep_every == 0:
            files_to_keep.append(file)
        else:
            files_to_delete.append(file)
    
    print(f"\nWill keep {len(files_to_keep)} files")
    print(f"Will delete {len(files_to_delete)} files")
    
    # Show some examples
    if files_to_keep:
        print(f"\nExample files to keep (first 5):")
        for file in files_to_keep[:5]:
            print(f"  - {file.name}")
    
    if files_to_delete and len(files_to_delete) > 0:
        print(f"\nExample files to delete (first 5):")
        for file in files_to_delete[:5]:
            print(f"  - {file.name}")
    
    # Confirm deletion
    if not dry_run and files_to_delete:
        response = input(f"\nAre you sure you want to delete {len(files_to_delete)} files? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Delete files
    deleted_count = 0
    for file in files_to_delete:
        if not dry_run:
            try:
                file.unlink()  # Delete the file
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file.name}: {e}")
        else:
            deleted_count += 1
    
    if dry_run:
        print(f"\nDRY RUN: Would have deleted {deleted_count} files")
        print("No files were actually deleted.")
    else:
        print(f"\nSuccessfully deleted {deleted_count} files")
        print(f"Kept {len(files_to_keep)} files (1 out of every {keep_every})")

if __name__ == "__main__":
    # Get directory from command line or use current directory
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.getcwd()
    
    # Optional: add dry-run flag
    dry_run = "--dry-run" in sys.argv or "-d" in sys.argv
    
    # Optional: specify different interval
    keep_every = 30
    for arg in sys.argv:
        if arg.startswith("--keep-every="):
            try:
                keep_every = int(arg.split("=")[1])
            except ValueError:
                print(f"Invalid value for --keep-every: {arg.split('=')[1]}")
                keep_every = 30
    
    print(f"Processing directory: {directory}")
    print(f"Keeping 1 out of every {keep_every} images")
    print(f"Dry run: {dry_run}")
    
    # Run the function
    reduce_dataset(directory, keep_every, dry_run)
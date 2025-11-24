"""
Dataset utilities - scanning, loading, and data manipulation
"""

from pathlib import Path
from PIL import Image


# Dataset paths relative to project root
DATASET_ROOT = Path(__file__).parent.parent.parent / "repo"
TRAIN_PATH = DATASET_ROOT / "training"
VAL_PATH = DATASET_ROOT / "validation"


def scan_dataset():
    """Scan repo/training and repo/validation directories"""
    dataset_info = {
        'train_samples': {},
        'val_samples': {},
        'classes': [],
        'total_train': 0,
        'total_val': 0,
        'sample_paths': {}
    }

    # Scan training directory
    if TRAIN_PATH.exists():
        for class_dir in sorted(TRAIN_PATH.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob("*.png"))
                class_name = class_dir.name
                dataset_info['train_samples'][class_name] = len(images)
                dataset_info['total_train'] += len(images)

                # Store sample paths for visualization
                if images:
                    dataset_info['sample_paths'][class_name] = images[:10]

    # Scan validation directory
    if VAL_PATH.exists():
        for class_dir in sorted(VAL_PATH.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob("*.png"))
                class_name = class_dir.name
                dataset_info['val_samples'][class_name] = len(images)
                dataset_info['total_val'] += len(images)

    # Get unique classes
    dataset_info['classes'] = sorted(
        set(dataset_info['train_samples'].keys()) |
        set(dataset_info['val_samples'].keys())
    )

    return dataset_info


def calculate_split_percentages(train_pct, val_of_remaining_pct):
    """
    Calculate final train/val/test percentages from 2 sliders

    Args:
        train_pct: Percentage for training (0-100)
        val_of_remaining_pct: Percentage of remaining data for validation (0-100)

    Returns:
        tuple: (train_pct, val_pct, test_pct)
    """
    remaining = 100 - train_pct
    val_pct = (remaining * val_of_remaining_pct) / 100
    test_pct = remaining - val_pct

    return train_pct, val_pct, test_pct


def get_image_dimensions(img_path):
    """Get dimensions of an image file"""
    try:
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except:
        return None

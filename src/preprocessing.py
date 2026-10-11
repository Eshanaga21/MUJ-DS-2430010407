"""
preprocessing.py — Data loading and pipeline for Waste Segregation

Scans for class folders, validates image readability, creates stratified splits,
and builds tf.data.Dataset pipelines with optional augmentation.
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


def set_seeds(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _is_valid_image(filepath):
    """Verify that an image file can be opened and read."""
    try:
        with Image.open(filepath) as im:
            im.verify()
        return True
    except Exception:
        return False


def load_dataset(root_path, verify_images=True):
    """
    Scan the dataset directory for class folders and return a DataFrame.

    Args:
        root_path: Path to the dataset root containing class subdirectories.
        verify_images: If True, check readability of each image file.

    Returns:
        df: DataFrame with columns ['filepath', 'class', 'class_id']
        class_names: list of class name strings
    """
    root = Path(root_path)
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    mapping = {name: i for i, name in enumerate(class_names)}

    rows = []
    for d in class_dirs:
        for f in d.rglob('*'):
            if f.is_file() and f.suffix.lower() in EXTS:
                if not verify_images or _is_valid_image(str(f)):
                    rows.append((str(f), d.name, mapping[d.name]))

    df = pd.DataFrame(rows, columns=['filepath', 'class', 'class_id'])
    return df, class_names


def create_splits(df, test_size=0.15, val_ratio=0.1765, seed=SEED):
    """
    Create stratified train/val/test splits (70/15/15) matching the notebook.

    Returns:
        train_df, val_df, test_df
    """
    dev_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df.class_id, random_state=seed,
    )
    train_df, val_df = train_test_split(
        dev_df, test_size=val_ratio, stratify=dev_df.class_id, random_state=seed,
    )
    return train_df, val_df, test_df


def _decode(path, label):
    """Decode, resize, and normalize a single image."""
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(frame, training=False, batch_size=BATCH_SIZE):
    """
    Build a tf.data.Dataset from a DataFrame split.

    Args:
        frame: DataFrame with 'filepath' and 'class_id' columns
        training: If True, shuffle the dataset
        batch_size: Batch size

    Returns:
        tf.data.Dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((
        frame.filepath.to_numpy(),
        frame.class_id.to_numpy(np.int32),
    ))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size).prefetch(AUTO)
    return ds


def compute_class_weights(train_df):
    """
    Compute class weights if the dataset is significantly imbalanced
    (ratio >= 1.5). Returns None if balanced.
    """
    counts = train_df.class_id.value_counts()
    ratio = counts.max() / counts.min()
    if ratio >= 1.5:
        weights = compute_class_weight(
            'balanced', classes=np.arange(2), y=train_df.class_id,
        )
        return {int(i): float(v) for i, v in enumerate(weights)}
    return None

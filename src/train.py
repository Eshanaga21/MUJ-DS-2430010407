"""
train.py — Training logic for Waste Segregation models

Trains a model with EarlyStopping and ReduceLROnPlateau callbacks.
Supports checkpointing to avoid re-training completed experiments.
"""

import gc
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train_or_load(name, builder_fn, train_ds, val_ds,
                  model_dir, result_dir,
                  epochs=30, class_weight=None):
    """
    Train a model or load from checkpoint if it already exists.

    Args:
        name: Experiment name (used for file naming)
        builder_fn: Callable that returns a compiled Keras model
        train_ds: Training tf.data.Dataset
        val_ds: Validation tf.data.Dataset
        model_dir: Directory to save/load model checkpoints
        result_dir: Directory to save training history CSVs
        epochs: Maximum number of training epochs
        class_weight: Optional dict of class weights

    Returns:
        model: Trained Keras model
        history: Training history dict
        seconds: Training time in seconds (0 if loaded from checkpoint)
    """
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{name}.keras'
    history_path = result_dir / f'{name}_history.csv'

    if model_path.exists():
        print(f'Loading checkpoint: {name}')
        model = keras.models.load_model(str(model_path))
        if history_path.exists():
            history = pd.read_csv(str(history_path)).to_dict(orient='list')
        else:
            history = {}
        return model, history, 0.0

    # Fresh training
    tf.keras.backend.clear_session()
    gc.collect()

    model = builder_fn()

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6,
        ),
    ]

    start = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    seconds = time.time() - start

    model.save(str(model_path))
    pd.DataFrame(hist.history).to_csv(str(history_path), index=False)

    return model, hist.history, seconds

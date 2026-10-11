"""
model.py — Model architectures for Waste Segregation

Contains:
  - plain_cnn(): SimpleCNN baseline (Flatten-based)
  - feature_aggregation_cnn(): RLFA-CNN with augmentation, BN, GAP, and optional L1
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

IMG_SIZE = (128, 128)


def get_augmentation_layer():
    """Moderate augmentation applied during training only."""
    return keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.10),
    ], name='moderate_augmentation')


def plain_cnn():
    """
    SimpleCNN baseline — uses Flatten + Dense.
    ~4.3M parameters. No augmentation, no BN, no GAP.
    """
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def feature_aggregation_cnn(use_l1=False, aug_layer=None):
    """
    RLFA-CNN (Regularized Lightweight Feature Aggregation CNN).
    ~111K parameters. Includes augmentation, BN, GAP, and optional L1.

    Args:
        use_l1: If True, applies L1 kernel regularization (lambda=1e-5)
        aug_layer: Augmentation layer to prepend (only active during training)
    """
    penalty = regularizers.l1(1e-5) if use_l1 else None

    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
    ])

    if aug_layer is not None:
        model.add(aug_layer)

    for filters in (32, 64, 128):
        model.add(layers.Conv2D(
            filters, 3, padding='same', activation='relu',
            kernel_regularizer=penalty,
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=penalty))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

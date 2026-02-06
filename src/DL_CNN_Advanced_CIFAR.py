# -*- coding: utf-8 -*-
"""
Implement CIFAR AdvancedCNN class that constructs, trains and evaluates a a more advanced CNN model in tensorflow.

Dataset: CIFAR-10 image dataset (available in Keras) that consists of 60,000 32 × 32 colour images in 10 classes.
We are going to use a subsetted version of this dataset to train some image classification models

CNN Model Architecture:

Layer (type)                 Output Shape              Param #
=================================================================
conv2d_9 (Conv2D)            (None, 32, 32, 32)        896

batch_normalization_3 (Batch (None, 32, 32, 32)        128
Normalization)

conv2d_10 (Conv2D)           (None, 30, 30, 32)        9248

max_pooling2d_5 (MaxPooling2 (None, 15, 15, 32)        0
D)

dropout_4 (Dropout)          (None, 15, 15, 32)        0

conv2d_11 (Conv2D)           (None, 15, 15, 64)        18496

batch_normalization_4 (Batch (None, 15, 15, 64)        256
Normalization)

conv2d_12 (Conv2D)           (None, 13, 13, 64)        36928

max_pooling2d_6 (MaxPooling2 (None, 6, 6, 64)          0
D)

dropout_5 (Dropout)          (None, 6, 6, 64)          0

conv2d_13 (Conv2D)           (None, 6, 6, 128)         73856

batch_normalization_5 (Batch (None, 6, 6, 128)         512
Normalization)

conv2d_14 (Conv2D)           (None, 4, 4, 128)         147584

max_pooling2d_7 (MaxPooling2 (None, 2, 2, 128)         0
D)

dropout_6 (Dropout)          (None, 2, 2, 128)         0

flatten_2 (Flatten)          (None, 512)               0

dense_4 (Dense)              (None, 128)               65664

dropout_7 (Dropout)          (None, 128)               0

dense_5 (Dense)              (None, 10)                1290

=================================================================

"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils
import numpy as np

class CIFAR_AdvancedCNN:
  def __init__(self, subset_size=5000):
    self.subset_size = subset_size
    self.model = None
    self.x_train = None
    self.y_train = None
    self.x_test = None
    self.y_test = None

  def load_and_preprocess_data(self):
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert class vectors to binary class matrices (One-hot encoding)
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    # Subset the data for faster training/experimentation
    self.x_train = x_train[:self.subset_size]
    self.y_train = y_train[:self.subset_size]
    self.x_test = x_test
    self.y_test = y_test

    print(f"Data loaded: {self.x_train.shape[0]} training samples")

  def define_model(self):
    self.model = models.Sequential()

    # --- 1 ---
    # Conv2D: 32 filters, same padding to keep 32x32 output
    self.model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    self.model.add(layers.BatchNormalization())

    # Conv2D: 32 filters, valid padding reduces 32x32 -> 30x30
    self.model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))

    # MaxPool: reduces 30x30 -> 15x15
    self.model.add(layers.MaxPooling2D((2, 2)))
    self.model.add(layers.Dropout(0.2)) # Common dropout rate for conv layers

    # --- 2 ---
    # Conv2D: 64 filters, same padding keeps 15x15
    self.model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    self.model.add(layers.BatchNormalization())

    # Conv2D: 64 filters, valid padding reduces 15x15 -> 13x13
    self.model.add(layers.Conv2D(64, (3, 3), padding='valid', activation='relu'))

    # MaxPool: reduces 13x13 -> 6x6
    self.model.add(layers.MaxPooling2D((2, 2)))
    self.model.add(layers.Dropout(0.3))

    # --- 3 ---
    # Conv2D: 128 filters, same padding keeps 6x6
    self.model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    self.model.add(layers.BatchNormalization())

    # Conv2D: 128 filters, valid padding reduces 6x6 -> 4x4
    self.model.add(layers.Conv2D(128, (3, 3), padding='valid', activation='relu'))

    # MaxPool: reduces 4x4 -> 2x2
    self.model.add(layers.MaxPooling2D((2, 2)))
    self.model.add(layers.Dropout(0.4))

    # --- Flatten & Dense ---
    self.model.add(layers.Flatten())

    # Dense: 128 neurons
    self.model.add(layers.Dense(128, activation='relu'))
    self.model.add(layers.Dropout(0.5)) # Higher dropout for dense layers

    # Output: 10 neurons
    self.model.add(layers.Dense(10, activation='softmax'))

    # Print summary to verify it matches the prompt's architecture
    self.model.summary()

  def compile_and_train_model(self, epochs=10):
    if self.model is None:
        raise ValueError("Model not defined")

    self.model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    history = self.model.fit(self.x_train, self.y_train,
                              epochs=epochs,
                              batch_size=64,
                              validation_split=0.1,
                              verbose=1)
    return history

  def evaluate_model(self):
    if self.model is None:
        raise ValueError("Model not trained.")

    test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss, test_acc

cifar_adv_cnn = CIFAR_AdvancedCNN(subset_size=5000)
cifar_adv_cnn.load_and_preprocess_data()
cifar_adv_cnn.define_model()
cifar_adv_cnn.compile_and_train_model(epochs=20) # Training for more epochs
test_loss, test_accuracy = cifar_adv_cnn.evaluate_model()
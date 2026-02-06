# -*- coding: utf-8 -*-

"""
Implement a CIFAR CNN class that constructs, trains and evaluates a simple CNN model in tensorflow

Dataset: CIFAR-10 image dataset (available in Keras) that consists of 60,000 32 × 32 colour images in 10 classes.
We are going to use a subsetted version of this dataset to train some image classification models

CNN Model Architecture:

Layer (type)                 Output Shape              Param #
=================================================================
conv2d_15 (Conv2D)           (None, 30, 30, 32)        896

max_pooling2d_8 (MaxPooling2D) (None, 15, 15, 32)      0

conv2d_16 (Conv2D)           (None, 13, 13, 64)        18496

max_pooling2d_9 (MaxPooling2D) (None, 6, 6, 64)        0

conv2d_17 (Conv2D)           (None, 4, 4, 64)          36928

flatten_3 (Flatten)          (None, 1024)              0

dense_6 (Dense)              (None, 64)                65600

dense_7 (Dense)              (None, 10)                650

=================================================================
Total params: 122570 (478.79 KB)
Trainable params: 122570 (478.79 KB)
Non-trainable params: 0 (0.00 Byte)

"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils

class CIFAR_CNN:
    def __init__(self, subset_size=5000):
      self.subset_size = subset_size
      self.model = None
      self.x_train = None
      self.y_train = None
      self.x_test = None
      self.y_test = None

    def load_and_preprocess_data(self):
      #Load CIFAR-10 data
      (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

      # Normalize pixel values to be between 0 and 1
      x_train, x_test = x_train / 255.0, x_test / 255.0
      # Convert class vectors to binary class matrices
      y_train = utils.to_categorical(y_train, 10)
      y_test = utils.to_categorical(y_test, 10)

      # Subset the data for faster training
      self.x_train = x_train[:self.subset_size]
      self.y_train = y_train[:self.subset_size]
      self.x_test = x_test[:self.subset_size] # Subsetting test too for consistency
      self.y_test = y_test[:self.subset_size]

      print(f"Data loaded: {self.x_train.shape[0]} training samples")

    def construct_model(self):
        # To match the output shapes exactly:
        # 32->30 requires 3x3 kernel with valid padding
        # 15->13 requires 3x3 kernel with valid padding
        # 6->4   requires 3x3 kernel with valid padding

        self.model = models.Sequential()

        # Layer 1: Conv2D (None, 30, 30, 32)
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

        # Layer 2: MaxPooling2D (None, 15, 15, 32)
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Layer 3: Conv2D (None, 13, 13, 64)
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # Layer 4: MaxPooling2D (None, 6, 6, 64)
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Layer 5: Conv2D (None, 4, 4, 64)
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # Layer 6: Flatten (None, 1024)
        self.model.add(layers.Flatten())

        # Layer 7: Dense (None, 64)
        self.model.add(layers.Dense(64, activation='relu'))

        # Layer 8: Dense (None, 10) - Output layer
        self.model.add(layers.Dense(10, activation='softmax'))

        # Print summary to verify it matches the prompt
        self.model.summary()

    def compile_and_train_model(self, epochs=10):
      if self.model is None:
            raise ValueError("Model not constructed.")

      self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

      history = self.model.fit(self.x_train, self.y_train,
                                epochs=epochs,
                                batch_size=64,
                                validation_split=0.1)
      return history

    def evaluate_model(self):
      if self.model is None:
             raise ValueError("Model not trained yet.")

      test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=2)
      print(f"\nTest accuracy: {test_acc:.4f}")
      return test_loss, test_acc

cifar_cnn = CIFAR_CNN(subset_size=5000)
cifar_cnn.load_and_preprocess_data()
cifar_cnn.construct_model()
cifar_cnn.compile_and_train_model(epochs=10)
test_loss, test_accuracy = cifar_cnn.evaluate_model()
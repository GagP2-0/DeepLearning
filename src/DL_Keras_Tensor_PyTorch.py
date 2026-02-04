# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
import torch
print(torch.__version__)
from sklearn.metrics import mean_squared_error, r2_score

# Create synthetic data for Linear Regression as per the question
np.random.seed(0)
x1 = np.random.rand(100,1) * 10 # 100 datapoints, uniform(0,1) * 10 for x1
x2 = np.random.rand(100,1) * 5 # 100 datapoints, uniform(0,1) * 5 for x2
true_weights = [2.0,1.5]
true_bias = 1.0
y = true_weights[0] * x1 + true_weights[1] * x2 + true_bias + np.random.randn(100,1) * 1.5

# Combine x1 and x2 into a single array
X = np.hstack((x1,x2))

"""
**********************************************************************************************************
1. LinearRegressorKeras class that constructs, trains and evaluates a single dense layer NN model in
Keras (e.g., Dense(units=1, input shape=(2,), activation=’linear’)).
***********************************************************************************************************
"""

class LinearRegressorKeras:
  def __init__(self):
    # Initialize the model
    self.model = None

  def __call__(self):
    # Construct the Keras Model
    self.model = tf.keras.Sequential([
      tf.keras.layers.Dense(units=1, input_shape=(2,), activation='linear')
    ])

    # Compile the model
    # Using Adam as it handles unscaled data beter than SGD
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')


  def train(self, x, y, epochs=100):
    if self.model is None:
      raise RuntimeError("Model not constructed")

    # Fit the model to the training data
    print(f"Training for {epochs} epochs...")
    history = self.model.fit(x, y, epochs=epochs, verbose=0)
    print("Training complete.")

    # Print loss at every 20 epochs
    for epoch in range(0, epochs, 20):
      loss = history.history['loss'][epoch]
      print(f'Epoch {epoch}, Loss: {loss}')

  def assess_performance(self, x, y):
    #Implement a method to make predictions such that the method returns rmse and r2
    if self.model is None:
      raise RuntimeError("Model not constructed")

    # 1. Generate predictions
    y_pred = self.model.predict(x)

    # 2. Calculate RMSE (Root Mean Squared Error)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # 3. Calculate R2 Score
    r2 = r2_score(y, y_pred)

    return rmse, r2

regressor = LinearRegressorKeras()
regressor() # construct the model
regressor.train(X,y, epochs = 200) # train the model
rmse, r2 = regressor.assess_performance(X,y) # assess (training) performance

print("-" * 30)
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print("-" * 30)

# Inspect learned weights to see if they match true_weights [2.0, 1.5] and bias 1.0
weights, bias = regressor.model.layers[0].get_weights()
print(f"Learned Weights: {weights.flatten()}")
print(f"Learned Bias: {bias.flatten()}")

# Visualization for summarising and comparison
y_pred = regressor.model.predict(X)

plt.figure(figsize=(12, 5))

# Plot 1: Predicted vs Actual
# Goal: Dots should follow the red dashed line
plt.subplot(1, 2, 1)
plt.scatter(y, y_pred, color='blue', alpha=0.6)
# Draw the perfect diagonal line
min_val = min(np.min(y), np.min(y_pred))
max_val = max(np.max(y), np.max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual Values (y)')
plt.ylabel('Predicted Values (y_pred)')
plt.title(f'Predicted vs Actual (R2={r2:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""
**************************************************************************************************************
2. Training an evaluation using TensorFlow
**************************************************************************************************************
"""

# Convert to Tensors
X_tensor = tf.cast(tf.constant(X), dtype=tf.float32)
y_tensor = tf.cast(tf.constant(y), dtype=tf.float32)

# LinearRegressionModel_tf class that constructs, trains and evaluates a linear regression model in tensorflow.
class LinearRegressionModel_tf(tf.keras.Model):
  def __init__(self):
    super(LinearRegressionModel_tf, self).__init__()

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    self.loss_fn = tf.keras.losses.MeanSquaredError()

    # Use add_weight to ensure Keras tracks these as trainable variables
    # Shape is (2, 1) because we have 2 features and 1 output
    self.w = self.add_weight(shape=(2,1), initializer="random_normal", trainable=True, name='weights')
    self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name='bias')

  def call(self, inputs):
    # Forward pass - x1*w1 + x2*w2 + bias or matrix multiplication (X dot W) + bias
    return tf.matmul(inputs, self.w) + self.b

  def train(self, X, y, epochs=100):

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Open a GradientTape to record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # 1. Forward pass (compute predictions) . This should invoke call method to perform the forward pass.
            predictions = self(X)

            # 2. Calculate the loss (Mean Squared Error)
            loss = self.loss_fn(y, predictions)

        # 3. Compute gradients of the loss with respect to w and b
        gradients = tape.gradient(loss, self.trainable_variables)

        # 4. Adjust parameters to reduce loss
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Print progress every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}: Loss: {loss.numpy():.4f}")

    print("Training complete.")

  def assess_performance(self, X, y):
    # Get predictions (returns tensor)
    y_pred_tensor = self(X)

    # Convert to numpy for sklearn
    y_pred = y_pred_tensor.numpy()
    y_true = y.numpy()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return rmse, r2


regressor_tf = LinearRegressionModel_tf()
regressor_tf.train(X_tensor,y_tensor,epochs=5) # train the model with the tensor version of the data
rmse, r2 = regressor_tf.assess_performance(X_tensor,y_tensor) # assess (training) performance

weights = regressor_tf.w.numpy()
bias = regressor_tf.b.numpy()

print("\n--- Results ---")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Learned Weights: {weights.flatten()} (True: [2.0, 1.5])")
print(f"Learned Bias: {bias} (True: 1.0)")

# --- Visualization Code ---
y_pred = regressor_tf(X_tensor).numpy()
y_true = y_tensor.numpy()

plt.figure(figsize=(10, 5))

# Predicted vs Actual
plt.scatter(y_true, y_pred, color='blue', alpha=0.6, label='Data Points')
min_val = min(np.min(y_true), np.min(y_pred))
max_val = max(np.max(y_true), np.max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit')

plt.title(f'TensorFlow Custom Loop: Predicted vs Actual (R²={r2:.3f})')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

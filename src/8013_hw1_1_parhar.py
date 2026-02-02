"""
1. Initialize the weight vector w and bias b to zero, choose a small positive constant γ as the margin, set
   the learning rate α, and specify the maximum number of iterations T .

2. For each iteration t until T is reached, perform the following steps:
    (a) Randomly shuffle the training examples.
    (b) For each training example (xi ∈ X , yi ∈ y):
        i. Calculate f = w · xi + b.
        ii. If yi = 1 and f < γ, then update w = w + α · xi and b = b + α.
        iii. If yi = 0 and f ≥ −γ, then update w = w − α · xi and b = b − α.
3. Return the final weight vector w and bias b.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# --- 1. The Model Class ---
class SimplePerceptron:
  def __init__(self, margin=0.1, learning_rate=0.01, max_iter=1000):
    self.margin = margin
    self.learning_rate = learning_rate
    self.max_iter = max_iter
    self.weights = None
    self.bias = 0

  # Use the URL to load data file.
  # Split it into X and Y with X being the input for training and Y as the output label.
  def load_data(self, data_file):
    data = pd.read_csv(data_file)
    X = data.iloc[:, :-1].values
    y_raw = data.iloc[:, -1].values

    # Initialize an empty array of the same size
    y = np.zeros(y_raw.shape)

    # Had to do this, as getting error.
    # Convert strings to integers strictly
    # This handles cases where data might be "Y", "Yes", 1, etc.
    for i, val in enumerate(y_raw):
        # Check if the value looks like a "Yes" or 1
        if str(val).strip().upper() in ['Y', 'YES', '1']:
            y[i] = 1
        else:
            y[i] = 0

    return X, y

  def fit(self, train_file):
    X, y = self.load_data(train_file)
    n_samples, n_features = X.shape

    # Initialize weights to zeros if not already set
    if self.weights is None:
        self.weights = np.zeros(n_features)
        self.bias = 0

    # Loop for T iterations
    for t in range(self.max_iter):
        # (a) Randomly shuffle the training examples
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # (b) Iterate over each example
        for i in range(n_samples):
            xi = X_shuffled[i]
            yi = y_shuffled[i]

            # i. Calculate f = w . xi + b
            f = np.dot(self.weights, xi) + self.bias

            # ii. If yi = 1 and f < gamma
            if yi == 1 and f < self.margin:
                self.weights = self.weights + (self.learning_rate * xi)
                self.bias = self.bias + self.learning_rate

            # iii. If yi = 0 and f >= -gamma
            elif yi == 0 and f >= -self.margin:
                self.weights = self.weights - (self.learning_rate * xi)
                self.bias = self.bias - self.learning_rate

  def predict(self, test_file):
    X, _ = self.load_data(test_file)
    # Calculate raw score
    f_scores = np.dot(X, self.weights) + self.bias

    # Standard perceptron prediction: 1 if score >= 0, else 0
    predictions = np.where(f_scores >= 0, 1, 0)
    return predictions

  def calculate_scores(self, test_file):
    # Implement the method to calculate accuracy, weighted F1, and macro F1 scores
    # calls the predict() method to generate predictions
    # Get true labels (y_true)
    _, y_true = self.load_data(test_file)

    # Get predicted labels (y_pred)
    y_pred = self.predict(test_file)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, weighted_f1, macro_f1

perceptron = SimplePerceptron(margin= 0.1, learning_rate=0.01, max_iter=1000)
perceptron.fit("https://raw.githubusercontent.com/GagP2-0/DeepLearning/refs/heads/main/data/q1_train_data1.csv")
accuracy, weighted_f1, macro_f1 = perceptron.calculate_scores("https://raw.githubusercontent.com/GagP2-0/DeepLearning/refs/heads/main/data/q1_test_data1.csv")
# Printing the performance metrics
print("accuracy: %.3f, macro_f1: %.3f, weighted_f1: %.3f" %(accuracy, macro_f1, weighted_f1))

# --- 2. The Plotting Logic ---
def generate_plot():
    perceptron = SimplePerceptron(margin=0.1, learning_rate=0.01)
    print("Training Model...")
    perceptron.fit("https://raw.githubusercontent.com/GagP2-0/DeepLearning/refs/heads/main/data/q1_train_data1.csv")

    acc, w_f1, m_f1 = perceptron.calculate_scores("https://raw.githubusercontent.com/GagP2-0/DeepLearning/refs/heads/main/data/q1_test_data1.csv")
    print("accuracy: %.3f, macro_f1: %.3f, weighted_f1: %.3f" %(acc, m_f1, w_f1))

    # B. Visualize the Scores (The Proof)
    X_test, y_test = perceptron.load_data("https://raw.githubusercontent.com/GagP2-0/DeepLearning/refs/heads/main/data/q1_test_data1.csv")

    # Calculate final confidence scores for the test set
    test_scores = np.dot(X_test, perceptron.weights) + perceptron.bias

    plt.figure(figsize=(10, 6))

    # Plot Class 0 (No)
    plt.scatter(np.where(y_test==0)[0], test_scores[y_test==0], 
                color='red', label='Class 0 (No)', alpha=0.6)

    # Plot Class 1 (Yes)
    plt.scatter(np.where(y_test==1)[0], test_scores[y_test==1], 
                color='blue', label='Class 1 (Yes)', alpha=0.6)

    # Draw Decision Boundary and Margins
    plt.axhline(0, color='black', linewidth=2, label='Decision Boundary (0)')
    plt.axhline(perceptron.margin, color='green', linestyle='--', label='Safety Margin')
    plt.axhline(-perceptron.margin, color='green', linestyle='--')

    plt.title("Model Confidence Scores on Test Data", fontsize=14)
    plt.xlabel("Test Data Sample ID")
    plt.ylabel("Perceptron Score (Confidence)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- Run it ---
generate_plot()

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 


def predict(X, W):
    return sigmoid(np.dot(X, W))

def train(X, y, W, learning_rate, epochs):
    for epoch in range(epochs):
        # Forward pass
        y_pred = predict(X, W)

        # Compute the error
        error = y_pred - y

        # Backpropagation
        dW = np.dot(X.T, error * y_pred * (1 - y_pred))

        # Update weights
        W -= learning_rate * dW

    return W
# Load the dataset
data = np.loadtxt('dataset.txt', delimiter=',') 
X = data[:, :-1]  # Features
y = data[:, -1]   # Labels
# Add bias term to the input features
X = np.hstack((X, np.ones((X.shape[0], 1))))    
# Initialize weights
W = np.random.rand(X.shape[1])
# Train the model   
learning_rate = 0.01
epochs = 1000
W = train(X, y, W, learning_rate, epochs)   
# Predict on the training data
y_pred = predict(X, W)
# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
x_values = np.array([X[:, 0].min(), X[:, 0].max()])
y_values = -(W[0] * x_values + W[2]) / W[1]  # Decision boundary
plt.plot(x_values, y_values, color='red')   
plt.xlabel('Feature 1')
plt.ylabel('Feature 2') 
plt.title('Neural Network Decision Boundary')
plt.show()  







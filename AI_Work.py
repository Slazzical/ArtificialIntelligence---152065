import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for consistent results
np.random.seed(42)

# Load the dataset
file_path = 'C:/Users/Lenovo/Desktop/Nairobi Office Price Ex.xlsx'
data = pd.read_excel(file_path)

# Extract the relevant columns
x = data['SIZE'].values  # Feature: office size
y = data['PRICE'].values  # Target: office price

# Function to calculate Mean Squared Error
def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) between true and predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function to update slope (m) and intercept (c)
def gradient_descent(x, y, m, c, learning_rate):
    """
    Perform one step of gradient descent on the parameters.
    """
    n = len(x)
    # Predict using current m and c
    y_pred = m * x + c
    # Compute gradients
    dm = (-2/n) * np.sum(x * (y - y_pred))
    dc = (-2/n) * np.sum(y - y_pred)
    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    return m, c

# Initial parameters with an adjusted learning rate and more epochs
m, c = np.random.randn(), np.random.randn()  # Random initialization for slope and intercept
learning_rate = 0.0001  # Slightly higher learning rate for convergence
epochs = 10  # More epochs for better convergence

# Training the model
errors = []  # To store the error at each epoch
for epoch in range(epochs):
    # Predict values based on current m and c
    y_pred = m * x + c
    # Calculate MSE
    error = mean_squared_error(y, y_pred)
    errors.append(error)
    # Print the error for the current epoch
    if (epoch+1) % 1 == 0 or epoch == 0:  # Print every 10 epochs for readability
        print(f"Epoch {epoch+1}/{epochs}, MSE: {error:.4f}")
    # Update m and c using gradient descent
    m, c = gradient_descent(x, y, m, c, learning_rate)

# Plotting the line of best fit after training
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, m * x + c, color='red', label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft)")
plt.ylabel("Office Price")
plt.title("Line of Best Fit for Office Size vs Price")
plt.legend()
plt.show()

# Prediction for an office size of 100 sq. ft
predicted_price_100 = m * 100 + c
print(f"Predicted price for 100 sq. ft: {predicted_price_100}")

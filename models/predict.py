import os
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the specific file names for test and cross-validation sets
train_file_name = "train_set.csv"
test_file_name = "test_set.csv"
cv_file_name = "cv_set.csv"

# Join the common path with the specific file names
test_file_path = os.path.join(common_path, test_file_name)
cv_file_path = os.path.join(common_path, cv_file_name)
train_file_path = os.path.join(common_path, train_file_name)

# Load the test set and cross-validation set into pandas DataFrames
test_data = pd.read_csv(test_file_path)
cv_data = pd.read_csv(cv_file_path)
train_data = pd.read_csv(train_file_path)

# Separate features (X) and target (y)
X = train_data.drop(columns=["salary_in_usd"])  # Drop the "salary_in_usd" column to get features
y = train_data["salary_in_usd"]  # Get the "salary_in_usd" column as the target
x_cv = cv_data.drop(columns=["salary_in_usd"])
y_cv = cv_data["salary_in_usd"]
x_test = test_data.drop(columns=["salary_in_usd"])
y_test = test_data["salary_in_usd"]

# Display the shapes of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
# X has 259 features!

# Create the model
model = Sequential ([
    Dense(units = 64, activation = 'relu'),
    Dense(units = 10, activation = 'relu'), #10 is the number of original features
    Dense(units = 1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the training data and store the history
history = model.fit(X, y, batch_size=32, epochs=100, validation_data=(x_cv, y_cv))

# Plot the epoch vs. the cost function
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

# Visualize the dataset
feature_column_index = 0
plt.scatter(X[:, feature_column_index], y, label='Training Data')
plt.xlabel('Feature')
plt.ylabel('Salary (USD)')
plt.legend()
plt.title('Dataset Visualization')
plt.show()








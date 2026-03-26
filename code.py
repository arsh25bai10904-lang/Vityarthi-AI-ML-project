import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- 1. Load Data ---
house_data = pd.read_csv("dataset.csv")

# --- 2. Feature Selection ---
features = ['sqft_living', 'bedrooms', 'bathrooms', 'grade', 'view']
X = house_data[features]

# Scaling the target variable (Price) for easier handling [00:03:18]
# This divides the large price numbers to make the model work faster
y = house_data['price'] / 100000 

# --- 3. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# --- 4. Build and Train the Model ---
model = LinearRegression()

# Training (Fitting) the model with training data 
model.fit(X_train, y_train)

# --- 5. Make Predictions ---
y_test_pred = model.predict(X_test)

# --- 6. Visualization ---
# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='black', label='Actual Price', s=10)
plt.scatter(range(len(y_test_pred)), y_test_pred, color='red', marker='*', label='Predicted Price', s=10)
plt.title("Comparison of Actual and Predicted Prices")
plt.xlabel("Data Points")
plt.ylabel("Price (Scaled)")
plt.legend()
plt.show() # In VS Code, you must use .show() to see the window

# --- 7. View Results Side-by-Side ---
# Creating a table to see the first 50 original vs predicted prices 
comparison_df = pd.DataFrame({'Original Price': y_test, 'Predicted Price': y_test_pred})
print("First 50 Comparisons:")
print(comparison_df.head(50))

# --- 8. Predict Price for a Custom House ---
# Example: 2000 sqft, 3 bedrooms, 2 bathrooms, Grade 7, View 0
custom_house = np.array([[2000, 3, 2, 7, 0]])
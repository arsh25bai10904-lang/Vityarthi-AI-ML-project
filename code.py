import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
house_data = pd.read_csv("dataset.csv")

features = ['sqft_living', 'bedrooms', 'bathrooms', 'grade', 'view']
X = house_data[features]

y = house_data['price'] / 100000 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='black', label='Actual Price', s=10)
plt.scatter(range(len(y_test_pred)), y_test_pred, color='red', marker='*', label='Predicted Price', s=10)
plt.title("Comparison of Actual and Predicted Prices")
plt.xlabel("Data Points")
plt.ylabel("Price (Scaled)")
plt.legend()
plt.show() 
comparison_df = pd.DataFrame({'Original Price': y_test, 'Predicted Price': y_test_pred})
print("First 50 Comparisons:")
print(comparison_df.head(50))

custom_house = np.array([[2000, 3, 2, 7, 0]])

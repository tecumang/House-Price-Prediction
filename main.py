import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your house price dataset (replace 'your_house_price_data.csv' with your actual file)
df = pd.read_csv('house_data.csv')

# Assuming 'SquareFootage', 'Bedrooms', and other relevant features in your dataset
X = df[['sqft_living', 'bedrooms', 'sqft_lot']]  # Add other features as needed
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualize predictions vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("House Price Prediction")
plt.show()

# Now, you can use the trained model to predict the price of a new house
new_house_features = [[new_square_footage, new_bedrooms, new_other_features]]
predicted_price = model.predict(new_house_features)
print(f"Predicted Price for the New House: {predicted_price[0]:.2f}")

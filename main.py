import pandas as pd

# Load FAO dataset
df = pd.read_csv("fao_data.csv")

# Keep only relevant columns
df = df[['Year', 'Item', 'Element', 'Value']]

# Pivot the data to make 'Element' as columns
df = df.pivot_table(index=['Year', 'Item'], columns='Element', values='Value').reset_index()

# Rename columns for clarity
df.columns = ['Year', 'Crop', 'Area_Harvested', 'Production', 'Yield']

# Drop rows with missing values (if any)
df = df.dropna()

# Display cleaned data
print("\nCleaned Data:")
print(df.head())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load and clean dataset
df = pd.read_csv("fao_data.csv")
df = df[['Year', 'Item', 'Element', 'Value']]
df = df.pivot_table(index=['Year', 'Item'], columns='Element', values='Value').reset_index()
df.columns = ['Year', 'Crop', 'Area_Harvested', 'Production', 'Yield']
df = df.dropna()

# Define input (X) and target (y)
X = df[['Area_Harvested', 'Production']]
y = df['Yield']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Generate predictions
y_pred = model.predict(X_test)

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle="dashed")  # 1:1 line
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.show()

# Line plot: Yield trends over years
sorted_indices = np.argsort(X_test['Year'].values.flatten())  # Sort by year
sorted_years = X_test['Year'].values.flatten()[sorted_indices]
sorted_actual = y_test.values.flatten()[sorted_indices]
sorted_predicted = y_pred.flatten()[sorted_indices]

plt.figure(figsize=(8, 6))
plt.plot(sorted_years, sorted_actual, marker='o', linestyle='-', label="Actual", color='blue')
plt.plot(sorted_years, sorted_predicted, marker='s', linestyle='--', label="Predicted", color='red')
plt.xlabel("Year")
plt.ylabel("Yield")
plt.title("Crop Yield Trends: Actual vs Predicted")
plt.legend()
plt.show()
import matplotlib.pyplot as plt

# Plot Actual vs Predicted Yield
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot(y_test, y_test, color='red', linestyle='dashed', label='Perfect Fit')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield')
plt.legend()
plt.show()

# Plot Yield Trends Over Time
plt.figure(figsize=(10, 5))
plt.plot(data['Year'], data['Yield'], label='Actual Yield', marker='o', linestyle='-')
plt.plot(data['Year'], model.predict(X), label='Predicted Yield', marker='s', linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Yield')
plt.title('Yield Trends Over Time')
plt.legend()
plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#  Load Dataset
#  Assuming that dataset has Temperature, Pressure, Catalyst, Yield
data = pd.read_csv("Reaction_data.csv")

#  Separating features(X) and target variable (y)
X = data.drop('Yield', axis=1)
y = data['Yield']

#  Splitting the data into Training and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Making Predictions on the test data
y_pred = model.predict(X_test_scaled)

# the mean squared error evaluating model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Feature Importance
feature_importance = model.feature_importances_
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_features = X.columns[sorted_indices]

# Visualize the data
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importance[sorted_indices])
plt.xticks(range(X.shape[1]), sorted_features, rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.tight_layout()
plt.show()

# Predict the yield for new conditions
new_conditions = np.array([[300, 5, 0.1]])
new_conditions_scaled = scaler.transform(new_conditions)
predicted_yield = model.predict(new_conditions_scaled)
print(f"Predicted Yield for New Conditions:{predicted_yield[0]:.2f}")

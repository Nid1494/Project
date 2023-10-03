import numpy as np
import pandas as pd

np.random.seed(42)

# Generate synthetic data
num_data_points = 60000
temperature = np.random.uniform(250, 450, num_data_points)
pressure = np.random.uniform(1, 15, num_data_points)
catalyst = np.random.uniform(0.05, 0.5, num_data_points)
noise = np.random.uniform(0, 5, num_data_points)
yield_data = 50 + 0.5 * temperature + 1.2 * pressure - 8 * catalyst + noise

# Create a DataFrame
data = pd.DataFrame({
    'Temperature': temperature,
    'Pressure': pressure,
    'Catalyst': catalyst,
    'Yield': yield_data
})

# Save the DataFrame to CSV file
data.to_csv('Reaction_data.csv', index=False)

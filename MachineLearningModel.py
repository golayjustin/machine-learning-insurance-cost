# Data Pre-Processing
# Import the Data
import pandas as pd

data = pd.read_csv('insurance.csv')

# Clean the Data
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

print(data.isnull().sum())

# Split Into Training and Test Sets
from sklearn.model_selection import train_test_split

x = data.drop('charges', axis=1)
y = data['charges']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modelling
# Build the Model
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Model
model.fit(x_train, y_train)

# Make Predictions
y_pred = model.predict(x_test)

# Evaluation
# Calculate Performance Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R2 Score: {r2}')

# Save the Model
import pickle

# Save the model to disk
filename = 'finalized_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# Load the model from disk
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)


#Step 1: Data Acquisition


#Load the Datasets

import pandas as pd

energy_data = pd.read_csv('World Energy Consumption.csv')

#Inspect Data

print(energy_data.head())

#Step 2: Define the methodology and the objectives of your work.

"""Specific Objectives:
Descriptive Analysis:

Analyze the historical energy consumption trends across different countries/regions.
Investigate the consumption of various energy sources (like oil, gas, coal, nuclear, and renewable energy).
Compare energy consumption trends to carbon emissions to study their impact on climate.
Predictive Analysis:

Energy Consumption Prediction: Develop a model to predict energy consumption based on historical data and key features (like population growth, GDP, energy source type).
Carbon Emission Prediction: Build a model to predict CO2 emissions based on energy consumption, which can help in estimating the environmental impact.
Policy Insight:

Identify patterns in how the transition to renewable energy could help reduce carbon emissions.
Provide insights on how countries/regions can shift their energy usage to cleaner sources to meet climate goals.
Methodology:
To achieve the objectives, the project will proceed as follows:

Exploratory Data Analysis (EDA):

Visualize the distribution of energy consumption globally and across different energy sources.
Analyze historical trends and correlation between energy consumption and CO2 emissions.
Feature Selection:

Select relevant features (e.g., year, country, energy source, GDP, population, etc.) that impact energy consumption and carbon emissions.
Data Preprocessing:

Handle missing data and scale features for effective modeling.
If there's class imbalance (e.g., certain energy sources or regions may have more data), use techniques like SMOTE to balance the data.
Modeling:

Use multiple ML models to predict future energy consumption and CO2 emissions:
Linear Regression: For establishing a baseline model.
Random Forest: To capture complex patterns in the data.
Time Series Models: Such as ARIMA or LSTM (if using temporal data).
Model Validation:

Use K-Fold Cross Validation to ensure robustness.
Evaluate models using metrics like mean squared error (MSE) for regression tasks, and R-squared to measure the model's accuracy in explaining variability.
Comparison and Interpretation:

Compare the performance of different models using suitable metrics.
Use visualization techniques to interpret how energy consumption changes over time and how it relates to carbon emissions."""

#Step 3: Data Preprocessing
# 1. Handling Missing Values
import pandas as pd

# Load the dataset
energy_data = pd.read_csv('World Energy Consumption.csv')

# Check for missing values
missing_values = energy_data.isnull().sum()

# Print out missing values
print(missing_values)

# Fill numeric columns with mean
numeric_cols = energy_data.select_dtypes(include=['float64', 'int64']).columns
energy_data[numeric_cols] = energy_data[numeric_cols].fillna(energy_data[numeric_cols].mean())

# Fill categorical columns with the mode (most frequent value)
categorical_cols = energy_data.select_dtypes(include=['object']).columns
energy_data[categorical_cols] = energy_data[categorical_cols].fillna(energy_data[categorical_cols].mode().iloc[0])

# After filling missing values, check again
print(energy_data.isnull().sum())

#2. Encoding Categorical Variables
# Apply one-hot encoding for categorical variables (like country, energy_source, etc.)
energy_data = pd.get_dummies(energy_data, columns=['country', 'year'], drop_first=True)

# Check the dataset after encoding
print(energy_data.head())

#3. Feature Scaling
from sklearn.preprocessing import StandardScaler

# Select numeric features to scale
numeric_features = energy_data.select_dtypes(include=['float64', 'int64']).columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling
energy_data[numeric_features] = scaler.fit_transform(energy_data[numeric_features])

# Check the scaled dataset
print(energy_data[numeric_features].head())

from imblearn.over_sampling import SMOTE
import numpy as np

# Load your dataset (replace 'your_file.csv' with the actual path to your dataset)
energy_data = pd.read_csv('World Energy Consumption.csv')

# Step 1: Handle missing values for numeric columns only
numeric_columns = energy_data.select_dtypes(include=[np.number]).columns
energy_data[numeric_columns] = energy_data[numeric_columns].fillna(energy_data[numeric_columns].mean())

# Step 2: Categorize 'coal_electricity' into discrete classes (based on estimated ranges)
energy_data['coal_category'] = pd.cut(
    energy_data['coal_electricity'],
    bins=[-1, 1000, 5000, np.inf],  # Adjust thresholds as needed
    labels=['low', 'medium', 'high']
)

# Define features (X) and the new categorical target (y)
X = energy_data.drop(columns=['coal_electricity', 'coal_category', 'population', 'year'])  # Drop unnecessary columns
y = energy_data['coal_category']  # Target is now the categorical column

# Convert categorical features to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 5: Check the new class distribution after applying SMOTE
print("After SMOTE, class distribution:\n", y_resampled.value_counts())


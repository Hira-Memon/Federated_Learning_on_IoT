import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import datetime

# Assuming you have a CSV file with the following columns:
# datetime, temperature, humidity, precip (1 for yes, 0 for no)
# Sample: 2023-01-15 14:30:00, 22.5, 85.3, 1

# Load the data
df = pd.read_csv('weather_data.csv')

# Convert datetime string to datetime object
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract features from datetime
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour
df['day_of_year'] = df['datetime'].dt.dayofyear

# Create season feature (1: Winter, 2: Spring, 3: Summer, 4: Fall)
def get_season(month):
    if month in [12, 1, 2]:
        return 1  # Winter
    elif month in [3, 4, 5]:
        return 2  # Spring
    elif month in [6, 7, 8]:
        return 3  # Summer
    else:
        return 4  # Fall

df['season'] = df['month'].apply(get_season)

# Select features and target
X = df[['temp', 'humidity', 'hour', 'day_of_year', 'season']]
y = df['precip']  # 1 for precipitation, 0 for no precipitation

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple decision tree model
# Using max_depth to limit tree complexity for Arduino implementation
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Export the tree as text for easy conversion to C/C++
tree_rules = export_text(model, feature_names=list(X.columns))
print("\nDecision Tree Rules:")
print(tree_rules)

# Now let's create a simple function to manually convert the decision tree to C code
print("\nConverting to Arduino code...")

# Extract feature thresholds from the tree
feature_indices = model.tree_.feature
thresholds = model.tree_.threshold
values = model.tree_.value
children_left = model.tree_.children_left
children_right = model.tree_.children_right

# Save the tree parameters to a file for Arduino implementation
np.savez('tree_params.npz', 
         feature_indices=feature_indices,
         thresholds=thresholds,
         values=values,
         children_left=children_left,
         children_right=children_right)

print("Done! Tree parameters saved to 'tree_params.npz'")
print("Use these values to implement the decision tree on your Arduino.")
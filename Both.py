import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
df = pd.read_csv("geopolitical_risk_data.csv")

# Encode categorical variables
label_encoders = {}
for col in ["Country", "Event_Type", "Risk_Category", "Source"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# Features (X) and Target (y)
X = df[["Country", "Event_Type", "Risk_Category", "Severity_Score", "Sentiment_Score"]]
y = df["Revenue_Loss%"]  # Predicting business revenue loss

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Model Performance:\n R² Score: {r2:.2f}\n MSE: {mse:.2f}\n RMSE: {rmse:.2f}")

# Example Prediction
example_event = np.array([[3, 2, 1, 75, -0.5]])  # Sample event data
predicted_loss = model.predict(example_event)
print(f"Predicted Revenue Loss: {predicted_loss[0]:.2f}%")

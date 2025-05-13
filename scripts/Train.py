import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("geopolitical_risk_data.csv")

# Encode categorical variables
label_encoders = {}
for col in ["Country", "Event_Type", "Risk_Category", "Source"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# Save encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Define features (X) and target (y)
X = df[["Country", "Event_Type", "Risk_Category", "Severity_Score", "Sentiment_Score"]]
y = df["Revenue_Loss%"]  # Predicting business revenue loss

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Model Performance:\n R² Score: {r2:.2f}\n MSE: {mse:.2f}\n RMSE: {rmse:.2f}")

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'model.pkl'")

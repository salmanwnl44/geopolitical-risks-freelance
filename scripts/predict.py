import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Function to encode categorical values
def encode_value(column, value):
    if column in label_encoders:
        return label_encoders[column].transform([value])[0]
    return value

# Example input: A geopolitical risk event
example_event = {
    "Country": "USA",
    "Event_Type": "Sanctions",
    "Risk_Category": "Economic",
    "Severity_Score": 85,
    "Sentiment_Score": -0.75
}

# Encode categorical values
encoded_event = [
    encode_value("Country", example_event["Country"]),
    encode_value("Event_Type", example_event["Event_Type"]),
    encode_value("Risk_Category", example_event["Risk_Category"]),
    example_event["Severity_Score"],
    example_event["Sentiment_Score"]
]

# Convert to NumPy array
example_event_array = np.array([encoded_event])

# Make prediction
predicted_loss = model.predict(example_event_array)

print(f"Predicted Revenue Loss: {predicted_loss[0]:.2f}%")

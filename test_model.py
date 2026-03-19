import joblib

# Load the saved model
model = joblib.load("food_model.pkl")

# Print the model type
print("Model loaded successfully!")
print(model)

# Make a sample prediction: [days_left, storage_type, temperature]
sample_data = [[5, 1, 5]] # Example: 5 days left, Fridge (1), 5 degrees
prediction = model.predict(sample_data)

print(f"Spoiled prediction (0=Fresh, 1=Spoiled): {prediction[0]}")

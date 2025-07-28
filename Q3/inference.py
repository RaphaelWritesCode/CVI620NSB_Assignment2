import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("mnist_model.pkl")
scaler = joblib.load("mnist_scaler.pkl")

# Load model type
with open("mnist_model_info.txt", "r") as f:
    model_type = f.read().strip()

def predict_digit(pixel_data):
    """
    Predict digit from pixel data
    
    Args:
        pixel_data: array of 784 pixel values (28x28 flattened)
    
    Returns:
        predicted digit (0-9)
    """
    
    # Normalize if not already done
    if pixel_data.max() > 1:
        pixel_data = pixel_data / 255.0
    
    # Reshape if needed
    if pixel_data.ndim == 1:
        pixel_data = pixel_data.reshape(1, -1)
    
    # Scale features
    pixel_data_scaled = scaler.transform(pixel_data)
    
    # Make prediction
    prediction = model.predict(pixel_data_scaled)[0]
    
    # Show confidence if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(pixel_data_scaled)[0]
        confidence = max(proba) * 100
        print(f"Predicted digit: {prediction} ({confidence:.1f}% confidence)")
    else:
        print(f"Predicted digit: {prediction}")
    
    return prediction

def predict_batch(pixel_batch):
    """Predict multiple digits at once"""
    
    # Normalize if not already done
    if pixel_batch.max() > 1:
        pixel_batch = pixel_batch / 255.0
    
    # Scale features
    pixel_batch_scaled = scaler.transform(pixel_batch)
    
    # Make predictions
    predictions = model.predict(pixel_batch_scaled)
    
    return predictions

# Example usage
if __name__ == "__main__":
    print(f"Loaded {model_type} model for MNIST digit classification")
    
    # Example: predict a random sample (replace with actual pixel data)
    random_pixels = np.random.randint(0, 256, 784)
    predict_digit(random_pixels)
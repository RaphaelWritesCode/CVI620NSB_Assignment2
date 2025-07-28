import cv2
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load model type
with open("model_info.txt", "r") as f:
    model_type = f.read().strip()

def predict_image(image_path):
    """Predict if image is cat (0) or dog (1)"""
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None

    img = cv2.resize(img, (64, 64))
    img = img.flatten() / 255.0
    img_scaled = scaler.transform(img.reshape(1, -1))

    # Make prediction
    prediction = model.predict(img_scaled)[0]
    label = "Cat" if prediction == 0 else "Dog"

    # Show confidence if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(img_scaled)[0]
        confidence = max(proba) * 100
        print(f"Prediction: {label} ({confidence:.1f}% confidence)")
    else:
        print(f"Prediction: {label}")

    return prediction

# Example usage
if __name__ == "__main__":
    print(f"Loaded {model_type} model")
    
    # Test with an example image
    test_image = r"C:\Users\rtape\Downloads\Seneca\CVI620NSB_Summer2025\codes\Assignment2\Q2\test\Cat\Cat (1).jpg"
    predict_image(test_image)
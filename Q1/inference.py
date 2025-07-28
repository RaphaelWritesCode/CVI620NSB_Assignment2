import numpy as np
import joblib

# Load models
lr_model = joblib.load('lr_model.pkl')
sgd_model = joblib.load('sgd_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_price(bedrooms, basement_area):
    """Predict house price using both models"""
    
    # Prepare input
    X = np.array([[bedrooms, basement_area]])
    
    # Linear Regression prediction
    lr_pred = lr_model.predict(X)[0]
    
    # SGD Regressor prediction (needs scaling)
    X_scaled = scaler.transform(X)
    sgd_pred = sgd_model.predict(X_scaled)[0]
    
    return lr_pred, sgd_pred

# Example usage
if __name__ == "__main__":
    bedrooms = 3
    basement_area = 2000
    
    lr_price, sgd_price = predict_price(bedrooms, basement_area)
    
    print(f"House: {bedrooms} bedrooms, {basement_area} sq ft")
    print(f"Linear Regression: ${lr_price:,.2f}")
    print(f"SGD Regressor: ${sgd_price:,.2f}")
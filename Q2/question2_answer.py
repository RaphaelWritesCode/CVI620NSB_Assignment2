import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Set your base directory
base_path = r"C:\Users\rtape\Downloads\Seneca\CVI620NSB_Summer2025\codes\Assignment2\Q2"

# Load training data
def load_train_data():
    data = []
    labels = []
    
    print("Loading training data...")

    # Load cat training images: cat.0.jpg to cat.4.jpg (you can increase range)
    for i in range(5):
        cat_path = os.path.join(base_path, "train", "Cat", f"cat.{i}.jpg")
        img = cv2.imread(cat_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img.flatten() / 255.0
            data.append(img)
            labels.append(0)
            print(f"Loaded: {cat_path}")
        else:
            print(f"Failed to load: {cat_path}")

    # Load dog training images: dog.0.jpg to dog.4.jpg
    for i in range(5):
        dog_path = os.path.join(base_path, "train", "Dog", f"dog.{i}.jpg")
        img = cv2.imread(dog_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img.flatten() / 255.0
            data.append(img)
            labels.append(1)
            print(f"Loaded: {dog_path}")
        else:
            print(f"Failed to load: {dog_path}")

    return np.array(data), np.array(labels)

# Load test data
def load_test_data():
    data = []
    labels = []

    print("Loading test data...")

    # Test cat images: Cat (1).jpg to Cat (5).jpg
    for i in range(1, 6):
        cat_path = os.path.join(base_path, "test", "Cat", f"Cat ({i}).jpg")
        img = cv2.imread(cat_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img.flatten() / 255.0
            data.append(img)
            labels.append(0)
            print(f"Loaded: {cat_path}")
        else:
            print(f"Failed to load: {cat_path}")

    # Test dog images: Dog (1).jpg to Dog (5).jpg
    for i in range(1, 6):
        dog_path = os.path.join(base_path, "test", "Dog", f"Dog ({i}).jpg")
        img = cv2.imread(dog_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img.flatten() / 255.0
            data.append(img)
            labels.append(1)
            print(f"Loaded: {dog_path}")
        else:
            print(f"Failed to load: {dog_path}")

    return np.array(data), np.array(labels)

# Load and preprocess data
X_train, y_train = load_train_data()
X_test, y_test = load_test_data()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Classifier
print("\nKNN CLASSIFIER")
print("-" * 30)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_pred)
print(f"KNN Accuracy: {knn_acc:.4f}")
print(classification_report(y_test, knn_pred, target_names=['Cat', 'Dog']))

# Logistic Regression
print("\nLOGISTIC REGRESSION")
print("-" * 30)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(classification_report(y_test, lr_pred, target_names=['Cat', 'Dog']))

# Model Comparison
print("\nMODEL COMPARISON")
print("-" * 30)
print(f"KNN Accuracy: {knn_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
model_name = "logistic_regression" if lr_acc >= knn_acc else "knn"
best_model = lr if model_name == "logistic_regression" else knn
print(f"Best Model: {model_name.replace('_', ' ').title()}")

# Save model
joblib.dump(best_model, f"{model_name}_model.pkl")
joblib.dump(scaler, f"{model_name}_scaler.pkl")
print(f"Model and scaler saved as {model_name}_model.pkl and {model_name}_scaler.pkl")

# Test on new image
def test_new_image(image_path):
    model = joblib.load(f'{model_name}_model.pkl')
    scaler = joblib.load(f'{model_name}_scaler.pkl')

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None

    img = cv2.resize(img, (64, 64))
    img = img.flatten() / 255.0
    img_scaled = scaler.transform(img.reshape(1, -1))

    prediction = model.predict(img_scaled)[0]
    label = "Cat" if prediction == 0 else "Dog"

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(img_scaled)[0]
        confidence = max(proba) * 100
        print(f"Prediction: {label} ({confidence:.1f}% confidence)")
    else:
        print(f"Prediction: {label}")

    return prediction

# Example usage
print("\nExample test command:")
print(r"test_new_image(r'C:\Users\rtape\Downloads\Seneca\CVI620NSB_Summer2025\codes\Assignment2\Q2\test\Cat\Cat (1).jpg')")

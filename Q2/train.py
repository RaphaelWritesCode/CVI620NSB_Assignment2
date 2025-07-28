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

def load_train_data():
    data = []
    labels = []
    
    print("Loading training data...")

    # Load cat training images: cat.0.jpg to cat.4.jpg
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

# Save best model
if lr_acc >= knn_acc:
    best_model = lr
    model_name = "logistic_regression"
    print("Best Model: Logistic Regression")
else:
    best_model = knn
    model_name = "knn"
    print("Best Model: KNN")

# Save model and scaler
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
with open("model_info.txt", "w") as f:
    f.write(model_name)

print(f"Model and scaler saved as best_model.pkl and scaler.pkl")
print(f"Model type: {model_name}")
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load MNIST data
print("Loading MNIST data...")
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# Split into features (X) and labels (y)
X_train = train_data.iloc[:, 1:].values / 255.0  # Normalize pixel values
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values / 255.0
y_test = test_data.iloc[:, 0].values

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Classifier
print("\nKNN CLASSIFIER")
print("-" * 30)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

knn_pred = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f"Accuracy: {knn_accuracy:.4f}")

# Logistic Regression
print("\nLOGISTIC REGRESSION")
print("-" * 30)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {lr_accuracy:.4f}")

# Model Comparison
print("\nMODEL COMPARISON")
print("-" * 30)
print(f"KNN Accuracy: {knn_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Check target achievement
if max(knn_accuracy, lr_accuracy) >= 0.90:
    print("✓ Target Achieved (≥ 90%)")
else:
    print("✗ Target Not Met (< 90%)")

# Save best model
if lr_accuracy >= knn_accuracy:
    best_model = lr
    model_name = "logistic_regression"
    print("Best Model: Logistic Regression")
else:
    best_model = knn
    model_name = "knn"
    print("Best Model: KNN")

# Save model and scaler
joblib.dump(best_model, "mnist_model.pkl")
joblib.dump(scaler, "mnist_scaler.pkl")
with open("mnist_model_info.txt", "w") as f:
    f.write(model_name)

print(f"Model saved as mnist_model.pkl")
print(f"Model type: {model_name}")
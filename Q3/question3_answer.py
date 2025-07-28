import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ===============================
# Step 1: Load and Prepare the Data
# ===============================

# Load MNIST training and testing data from CSV files
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# Split into features (X) and labels (y)
X_train = train_data.iloc[:, 1:].values / 255.0  # Normalize pixel values to [0, 1]
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values / 255.0
y_test = test_data.iloc[:, 0].values

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ===============================
# Step 2: Feature Scaling
# ===============================

# Standardize features: mean = 0, std = 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Step 3: KNN Classifier
# ===============================

print("\nKNN CLASSIFIER")
print("-" * 30)

# Initialize and train K-Nearest Neighbors (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Predict on test data and evaluate accuracy
knn_pred = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f"Accuracy: {knn_accuracy:.4f}")

# ===============================
# Step 4: Logistic Regression
# ===============================

print("\nLOGISTIC REGRESSION")
print("-" * 30)

# Initialize and train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Predict on test data and evaluate accuracy
lr_pred = lr.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {lr_accuracy:.4f}")

# ===============================
# Step 5: Model Comparison
# ===============================

print("\nMODEL COMPARISON")
print("-" * 30)
print(f"KNN Accuracy:              {knn_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Check if either model achieved ≥ 90% accuracy
if max(knn_accuracy, lr_accuracy) >= 0.90:
    print("✓ Target Achieved (≥ 90%)")
else:
    print("✗ Target Not Met (< 90%)")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# ===============================
# Step 1: Load and Prepare Data
# ===============================

# Load housing dataset
df = pd.read_csv('house_price.csv')

# Select features (size, bedroom) and target (price)
X = df[['size', 'bedroom']]
y = df['price']

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# Step 2: Linear Regression Model
# ===============================

print("LINEAR REGRESSION")
print("-" * 40)

# Initialize and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Display learned coefficients
coeff_df = pd.DataFrame(lr_model.coef_, X.columns, columns=['Coefficient'])
print("Coefficients:")
print(coeff_df)
print(f"Intercept: {lr_model.intercept_:.2f}")

# Predict on test data and evaluate performance
y_pred_lr = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_mape = mean_absolute_percentage_error(y_test, y_pred_lr)

print("\nMetrics:")
print(f"MAE:  {lr_mae:.2f}")
print(f"MSE:  {lr_mse:.2f}")
print(f"RMSE: {lr_rmse:.2f}")
print(f"MAPE: {lr_mape:.4f} ({lr_mape*100:.2f}%)")

# ===============================
# Step 3: SGD Regressor Model
# ===============================

print("\n\nSGD REGRESSOR")
print("-" * 40)

# Standardize features (required for gradient-based models like SGD)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train SGD Regressor
sgd_model = SGDRegressor(max_iter=1000, random_state=42)
sgd_model.fit(X_train_scaled, y_train)

# Display learned coefficients (note: scaled features)
sgd_coeff_df = pd.DataFrame(sgd_model.coef_, X.columns, columns=['Coefficient'])
print("Coefficients (scaled features):")
print(sgd_coeff_df)
print(f"Intercept: {sgd_model.intercept_[0]:.2f}")

# Predict and evaluate performance
y_pred_sgd = sgd_model.predict(X_test_scaled)
sgd_mae = mean_absolute_error(y_test, y_pred_sgd)
sgd_mse = mean_squared_error(y_test, y_pred_sgd)
sgd_rmse = np.sqrt(sgd_mse)
sgd_mape = mean_absolute_percentage_error(y_test, y_pred_sgd)

print("\nMetrics:")
print(f"MAE:  {sgd_mae:.2f}")
print(f"MSE:  {sgd_mse:.2f}")
print(f"RMSE: {sgd_rmse:.2f}")
print(f"MAPE: {sgd_mape:.4f} ({sgd_mape*100:.2f}%)")

# ===============================
# Step 4: Model Comparison
# ===============================

print("\n\nMODEL COMPARISON")
print("-" * 40)

# Create a comparison table of evaluation metrics
comparison = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE'],
    'LinearRegression': [lr_mae, lr_mse, lr_rmse, lr_mape],
    'SGDRegressor': [sgd_mae, sgd_mse, sgd_rmse, sgd_mape]
})
print(comparison.round(4))

# ===============================
# Step 5: Metrics Explanation
# ===============================

print("\n\nMETRICS TRADE-OFFS")
print("-" * 40)
print("MAE  - Average absolute error, robust to outliers.")
print("MSE  - Penalizes large errors more due to squaring.")
print("RMSE - Same units as target; balances MAE and MSE.")
print("MAPE - Scale-independent; expresses error in %.")

print(f"\nNote: RMSE is often preferred because it:")
print("- Uses the same unit as the target variable")
print("- Penalizes larger errors more than MAE")
print("- Is more interpretable than MSE in most cases")

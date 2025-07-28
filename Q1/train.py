import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Load data
df = pd.read_csv(r"C:\Users\rtape\Downloads\Seneca\CVI620NSB_Summer2025\codes\Assignment2\Q1\house_price.csv")
X = df[['bedroom', 'size']]  # bedrooms and basement area
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
print("LINEAR REGRESSION")
print("-" * 30)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Coefficients:")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {lr_model.intercept_:.2f}")

y_pred_lr = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_mape = mean_absolute_percentage_error(y_test, y_pred_lr)

print(f"MAE: {lr_mae:.2f}")
print(f"MSE: {lr_mse:.2f}")
print(f"RMSE: {lr_rmse:.2f}")
print(f"MAPE: {lr_mape:.4f}")

# SGD Regressor
print("\nSGD REGRESSOR")
print("-" * 30)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd_model = SGDRegressor(max_iter=1000, random_state=42)
sgd_model.fit(X_train_scaled, y_train)

print("Coefficients (scaled):")
for feature, coef in zip(X.columns, sgd_model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {sgd_model.intercept_[0]:.2f}")

y_pred_sgd = sgd_model.predict(X_test_scaled)
sgd_mae = mean_absolute_error(y_test, y_pred_sgd)
sgd_mse = mean_squared_error(y_test, y_pred_sgd)
sgd_rmse = np.sqrt(sgd_mse)
sgd_mape = mean_absolute_percentage_error(y_test, y_pred_sgd)

print(f"MAE: {sgd_mae:.2f}")
print(f"MSE: {sgd_mse:.2f}")
print(f"RMSE: {sgd_rmse:.2f}")
print(f"MAPE: {sgd_mape:.4f}")

# Comparison
print("\nCOMPARISON")
print("-" * 30)
comparison = pd.DataFrame({
    'LinearRegression': [lr_mae, lr_mse, lr_rmse, lr_mape],
    'SGDRegressor': [sgd_mae, sgd_mse, sgd_rmse, sgd_mape]
}, index=['MAE', 'MSE', 'RMSE', 'MAPE'])
print(comparison.round(4))

# Metrics explanation
print("\nMETRICS TRADE-OFFS:")
print("MAE: Less sensitive to outliers, same units as target")
print("MSE: Penalizes large errors more, different units (squared)")
print("RMSE: Same units as target, penalizes large errors")
print("MAPE: Scale-independent, percentage-based")

# Save models
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(sgd_model, 'sgd_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModels saved.")
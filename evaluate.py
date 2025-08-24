import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load predictions
results = pd.read_csv("../artifacts/model_predictions.csv")

y_true = results["y_test"]
y_pred_lr = results["y_pred_lr"]
y_pred_rf = results["y_pred_rf"]

# Linear Regression metrics
mse_lr = mean_squared_error(y_true, y_pred_lr)
r2_lr = r2_score(y_true, y_pred_lr)

# Random Forest metrics
mse_rf = mean_squared_error(y_true, y_pred_rf)
r2_rf = r2_score(y_true, y_pred_rf)

print("---- Model Evaluation ----")
print(f"Linear Regression -> MSE: {mse_lr:.2f}, R²: {r2_lr:.4f}")
print(f"Random Forest     -> MSE: {mse_rf:.2f}, R²: {r2_rf:.4f}")

# Save results to a file
with open("../artifacts/evaluation.txt", "w") as f:
    f.write("---- Model Evaluation ----\n")
    f.write(f"Linear Regression -> MSE: {mse_lr:.2f}, R²: {r2_lr:.4f}\n")
    f.write(f"Random Forest     -> MSE: {mse_rf:.2f}, R²: {r2_rf:.4f}\n")
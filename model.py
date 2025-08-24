import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("../artifacts/feature_data.csv")

X = df.drop(columns=['sold'])
y = df['sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Save models
joblib.dump(lr, "../artifacts/linear_model.pkl")
joblib.dump(rf, "../artifacts/rf_model.pkl")

# Save predictions
pd.DataFrame({
    "y_test": y_test,
    "y_pred_lr": y_pred_lr,
    "y_pred_rf": y_pred_rf
}).to_csv("../artifacts/model_predictions.csv", index=False)
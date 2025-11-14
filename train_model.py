import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("weather_data.csv")

# Choose target column
target_column = "temperature"
X = df.drop(columns=[target_column])
y = df[target_column]

# Identify feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Model - Random Forest (best general-purpose regressor)
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)

pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Trained Successfully")
print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/best_weather_regressor.pkl")
print("💾 Model saved to models/best_weather_regressor.pkl")

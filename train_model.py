import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# 1. LOAD & CLEAN
file_name = 'student_performance.csv'
if not os.path.exists(file_name):
    print(f"❌ Error: {file_name} not found! Please check your folder.")
    exit()

df = pd.read_csv(file_name)
df.columns = df.columns.str.strip() 

# 2. PRE-PROCESSING (Encoding text to numbers)
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MODEL EVALUATION (The Competition)
print("--- 🏁 Step 1: Evaluating Multiple Models ---")
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVM": SVR()
}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"{name} | Avg Training R2 Score: {cv_scores.mean():.4f}")

# 4. HYPERPARAMETER TUNING (Validation Phase)
print("\n--- ⚙️ Step 2: Tuning the Best Model (Random Forest) ---")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# 5. FINAL VALIDATION
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print("\n--- ✅ Step 3: Final Results (Validation) ---")
print(f"Final Tuned R2 Score: {r2_score(y_test, predictions):.4f}")
print(f"Average Prediction Error (MAE): {mean_absolute_error(y_test, predictions):.2f}")

# 6. SAVE EVERYTHING
save_data = {'model': best_model, 'features': feature_names}
joblib.dump(save_data, 'best_student_model.pkl')
print("\n🔥 Success! Optimized model saved to 'best_student_model.pkl'")
import pandas as pd
import numpy as np

# 1. Load the data
file_name = 'student_performance.csv'
df = pd.read_csv(file_name)

print("--- Data Before Cleaning ---")
print(df.head())
print(f"Initial Shape: {df.shape}")

# 2. Clean Column Names
# Removes hidden spaces like ' Score ' -> 'Score'
df.columns = df.columns.str.strip()

# 3. Handle Missing Values
# For numeric columns (Scores, Hours), we fill with the average (mean)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# For categorical columns (Gender, School), we fill with the most common value (mode)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Standardize Text Data
# Fixes "Male", "male ", and "MALE" to all be the same
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.capitalize()

# 5. Remove Duplicates
df = df.drop_duplicates()

# 6. Remove Outliers (Impossible Values)
# Example: If your marks are out of 100, remove anything higher
target_col = df.columns[-1]
df = df[(df[target_col] >= 0) & (df[target_col] <= 100)]

# 7. Save the Cleaned File
df.to_csv('student_performance_cleaned.csv', index=False)

print("\n--- Data After Cleaning ---")
print(f"Final Shape: {df.shape}")
print("✅ Success: 'student_performance_cleaned.csv' has been created.")

import pandas as pd
import numpy as np  # <--- THIS WAS MISSING
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
# Make sure the filename matches your actual file exactly
try:
    df = pd.read_csv('student_performance.csv')
    df.columns = df.columns.str.strip() # Remove hidden spaces

    # 2. Basic Inspection
    print("--- Summary Statistics ---")
    print(df.describe())

    # 3. Visualizations
    plt.figure(figsize=(12, 10))

    # A. Distribution of Scores (Histogram)
    plt.subplot(2, 2, 1)
    # iloc[:, -1] selects the last column automatically
    sns.histplot(df.iloc[:, -1], kde=True, color='blue')
    plt.title('Distribution of Final Scores')

    # B. Correlation Heatmap
    plt.subplot(2, 2, 2)
    # Select only numeric columns using the fixed 'np'
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')

    # C. Study Hours vs Score (Scatter Plot)
    # Automatically finds the relationship between the first and last numeric column
    plt.subplot(2, 2, 3)
    sns.scatterplot(x=numeric_df.iloc[:, 0], y=numeric_df.iloc[:, -1])
    plt.title(f'{numeric_df.columns[0]} vs {numeric_df.columns[-1]}')

    # D. Categorical Count (Bar Plot)
    plt.subplot(2, 2, 4)
    # Finds the first text column (like Gender or School)
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        sns.countplot(x=df[cat_cols[0]])
        plt.title(f'Count of {cat_cols[0]}')

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: 'student_performance.csv' not found. Ensure it is in the same folder.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------------- LOAD DATA ----------------
#drive.mount('/content/drive')

file_path = 'data8August2025.csv'
df = pd.read_csv(file_path)
print("Printing of Full data")
print(df)
# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Display dataset info
print("First 5 rows of the dataset:")
display(df.head())
print("\nShape after loading:", df.shape)

# Drop unnecessary columns
columns_to_drop = [
    'institution_name',
    'email_address',
    'timestamp',
    'institution_type',
    'home_district',
    'subject',
    'body_weight_level',
    'name'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
print("Columns after dropping:")
display(df.columns)
print("\nShape after dropping columns:", df.shape)


# Convert categorical strings to integer codes
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

# Data types after conversion
print("\nData types after categorical encoding:")
display(df.dtypes)

# Missing values
print("\nMissing values per column:")
display(df.isnull().sum())

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


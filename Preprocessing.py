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

# ---------------- FEATURE CORRELATION BEFORE SMOTE ----------------
print("\n=== Feature Correlation Analysis Before SMOTE ===")
# Calculate correlation matrix
corr_before = df.corr()

# Plot correlation heatmap
plt.figure(figsize=(24, 16))
mask = np.triu(np.ones_like(corr_before, dtype=bool))
sns.heatmap(
    corr_before,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5},
    annot_kws={"size": 8}  # font size of annotations
)
plt.title('Feature Correlation Matrix Before SMOTE', fontsize=16, pad=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.savefig("Figure Feature Correlation Before SMOTE.pdf", dpi=100)
plt.show()
target_col = 'do_you_feel_insecure_about_your_current_job_or_future_career?'
# ---------------- OLS REGRESSION BEFORE SMOTE ----------------
print("\n=== OLS Regression Model Before SMOTE ===")
# Prepare data for OLS
X_before = df.drop(target_col, axis=1)
y_before = df[target_col]

# Summary statistics
print("\nSummary statistics raw(Before SMOTE):")
display(X_before.describe())

# Add constant for OLS
X_before_const = sm.add_constant(X_before)

# Fit OLS model
ols_before = sm.OLS(y_before, X_before_const).fit()

# Print OLS summary
print("OLS Regression Summary Before SMOTE:")
print(ols_before.summary())

# Plot OLS coefficients
plt.figure(figsize=(12, 8))
coef_before = ols_before.params.drop('const').sort_values(ascending=False)
ax = sns.barplot(x=coef_before.values, y=coef_before.index, palette="viridis")
plt.title('OLS Regression Coefficients Before SMOTE', fontsize=16, pad=20)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Add value labels on the bars
for i, v in enumerate(coef_before.values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("Figure OLS Coefficients Before SMOTE.pdf", dpi=100)
plt.show()

# ---------------- TARGET SPLIT ----------------
target_col = 'do_you_feel_insecure_about_your_current_job_or_future_career?'
X = df.drop(target_col, axis=1)
y = df[target_col]

# Class distribution before SMOTE
print("\nOriginal class distribution:")
print(y.value_counts())

# Create a pie chart with different colors
plt.figure(figsize=(8, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(y.value_counts())))
explode = [0.05] * len(y.value_counts())  # Slightly explode all slices

# Define a function to format the labels with both percentage and count
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val})'
    return my_format

plt.pie(y.value_counts(),
        labels=y.value_counts().index,
        autopct=autopct_format(y.value_counts()),
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 12})

# Add a white circle in the middle to create a donut chart (optional)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Original Class Distribution", fontsize=16, pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.draw()
plt.savefig("Figure Original Class Distribution.pdf", dpi=100)
plt.show()

# t-SNE visualization before SMOTE
print("\nGenerating t-SNE visualization before SMOTE...")
plt.figure(figsize=(8, 6))

# Apply t-SNE to the original data (before SMOTE)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

# Create a scatter plot with different colors for each class
unique_classes = y.unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

for i, class_label in enumerate(unique_classes):
    plt.scatter(X_tsne[y == class_label, 0],
                X_tsne[y == class_label, 1],
                c=[colors[i]],
                label=f'Class {class_label}',
                alpha=0.7,
                s=30)

plt.title('t-SNE Visualization Before SMOTE', fontsize=16, pad=20)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Figure t-SNE Before SMOTE.pdf", dpi=100)
plt.show()

# ---------------- APPLY SMOTE ----------------
desired_count = 500
class_counts = y.value_counts()
sampling_strategy = {cls: desired_count for cls in class_counts.index if class_counts[cls] < desired_count}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nNew class distribution after SMOTE:")
print(y_resampled.value_counts())

plt.figure(figsize=(8, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(y_resampled.value_counts())))
explode = [0.05] * len(y_resampled.value_counts())  # Slightly explode all slices

# Define a function to format the labels with both percentage and count
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val})'
    return my_format

plt.pie(y_resampled.value_counts(),
        labels=y_resampled.value_counts().index,
        autopct=autopct_format(y_resampled.value_counts()),
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 12})

# Add a white circle in the middle to create a donut chart
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Class Distribution After SMOTE", fontsize=16, pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.draw()
plt.savefig("Figure Class Distribution After SMOTE.pdf", dpi=100)
plt.show()

# Combine resampled data
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# ---------------- FEATURE CORRELATION AFTER SMOTE ----------------
print("\n=== Feature Correlation Analysis After SMOTE ===")
# Calculate correlation matrix
corr_after = df_resampled.corr()

# Plot correlation heatmap
plt.figure(figsize=(24, 16))
mask = np.triu(np.ones_like(corr_after, dtype=bool))

sns.heatmap(
    corr_after,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5},
    annot_kws={"size": 8}  # Adjust font size here
)
plt.title('Feature Correlation Matrix After SMOTE', fontsize=16, pad=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.savefig("Figure Feature Correlation After SMOTE.pdf", dpi=100)
plt.show()

# ---------------- OLS REGRESSION AFTER SMOTE ----------------
print("\n=== OLS Regression Model After SMOTE ===")
# Prepare data for OLS
X_after = df_resampled.drop(target_col, axis=1)
y_after = df_resampled[target_col]

# Add constant for OLS
X_after_const = sm.add_constant(X_after)

# Fit OLS model
ols_after = sm.OLS(y_after, X_after_const).fit()

# Print OLS summary
print("OLS Regression Summary After SMOTE:")
print(ols_after.summary())

# Plot OLS coefficients
plt.figure(figsize=(12, 8))
coef_after = ols_after.params.drop('const').sort_values(ascending=False)
ax = sns.barplot(x=coef_after.values, y=coef_after.index, palette="viridis")
plt.title('OLS Regression Coefficients After SMOTE', fontsize=16, pad=20)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Add value labels on the bars
for i, v in enumerate(coef_after.values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("Figure OLS Coefficients After SMOTE.pdf", dpi=100)
plt.show()

# Create a visualization of descriptive statistics after SMOTE
print("\nSummary statistics (After SMOTE):")
display(X_resampled.describe())

# Create a heatmap of the descriptive statistics
plt.figure(figsize=(12, 8))
desc_stats = X_resampled.describe()
sns.heatmap(desc_stats, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True,
            linewidths=0.5, linecolor='white')
plt.title('Descriptive Statistics After SMOTE', fontsize=16, pad=20)
plt.xlabel('Statistics', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig("Figure Descriptive Statistics After SMOTE.pdf", dpi=100)
plt.show()

# Create a separate visualization for mean values
plt.figure(figsize=(10, 6))
mean_values = X_resampled.mean().sort_values(ascending=False)
ax = sns.barplot(x=mean_values.values, y=mean_values.index, palette="viridis")
plt.title('Mean Values of Features After SMOTE', fontsize=16, pad=20)
plt.xlabel('Mean Value', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Add value labels on the bars
for i, v in enumerate(mean_values.values):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("Figure Mean Values After SMOTE.pdf", dpi=100)
plt.show()

# Create a separate visualization for standard deviation
plt.figure(figsize=(10, 6))
std_values = X_resampled.std().sort_values(ascending=False)
ax = sns.barplot(x=std_values.values, y=std_values.index, palette="plasma")
plt.title('Standard Deviation of Features After SMOTE', fontsize=16, pad=20)
plt.xlabel('Standard Deviation', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Add value labels on the bars
for i, v in enumerate(std_values.values):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("Figure Standard Deviation After SMOTE.pdf", dpi=100)
plt.show()




# ---------------- T-SNE VISUALIZATION ----------------
print("\nRunning t-SNE... (may take some time)")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_embedded = tsne.fit_transform(X_resampled)

tsne_df = pd.DataFrame({
    'TSNE1': X_embedded[:,0],
    'TSNE2': X_embedded[:,1],
    'Class': y_resampled
})

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='TSNE1', y='TSNE2',
    hue='Class',
    palette='tab10',
    data=tsne_df,
    alpha=0.7
)
plt.title("t-SNE Visualization of Resampled Data")
plt.legend(title="Class")
plt.tight_layout()
plt.draw()
plt.savefig("Figure t-SNE Visualization of Resampled Data.pdf", dpi=100)
plt.show()

# ---------------- TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split( X_resampled, y_resampled, test_size=0.2, random_state=42 )
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


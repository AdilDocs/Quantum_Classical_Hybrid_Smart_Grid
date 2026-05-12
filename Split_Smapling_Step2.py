import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

df = pd.read_csv("time_series_clean.csv")


features = ['Load (MW)', 'Solar Generation (MW)', 'Price ($/MWh)', 's1_Gen', 's2_Line', 's3_XFMR']
X = df[features]
y = df['Label_y']

if X.isnull().values.any():
    print("Warning: NaNs detected in features. Dropping rows with missing values...")
    X = X.dropna()
    y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("-" * 30)
print("SPLIT EXECUTION COMPLETED")
print("-" * 30)
print(f"Original Train Set: {len(y_train)} scenarios")
print(f"  - Reliable (1): {Counter(y_train)[1]}")
print(f"  - Extreme (0):  {Counter(y_train)[0]}")

print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("-" * 30)
print("POST-SMOTE STATISTICS (TRAINING ONLY)")
print("-" * 30)
print(f"Resampled Train Set Size: {len(y_train_resampled)}")
print(f"New Class Distribution:   {Counter(y_train_resampled)}")
print("-" * 30)

X_train_resampled.to_csv("X_train_final.csv", index=False)
y_train_resampled.to_csv("y_train_final.csv", index=False)
X_test.to_csv("X_test_final.csv", index=False)
y_test.to_csv("y_test_final.csv", index=False)

print("Files saved successfully. Ready for Step 3.")
# ==========================================================
# STEP 15 - FEATURE IMPORTANCE USING XGBOOST
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ==========================================================
# STEP 1 - LOAD DATASET
# ==========================================================

print("="*60)
print("STEP 1 - LOADING DATASET")
print("="*60)

df = pd.read_csv("dataset/student_dataset_feature_engineered.csv")

print(df.head())

# ==========================================================
# STEP 2 - FEATURES AND TARGET
# ==========================================================

print("\n" + "="*60)
print("STEP 2 - FEATURES AND TARGET")
print("="*60)

X = df.drop(["StudentID", "Placement_Status"], axis=1)

# Convert categorical columns into numbers
X = pd.get_dummies(X)

y = df["Placement_Status"]

print("Features Shape :", X.shape)
print("Target Shape   :", y.shape)

# ==========================================================
# STEP 3 - TRAIN TEST SPLIT
# ==========================================================

print("\n" + "="*60)
print("STEP 3 - TRAIN TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# STEP 4 - CREATE XGBOOST MODEL
# ==========================================================

print("\n" + "="*60)
print("STEP 4 - TRAINING XGBOOST")
print("="*60)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# ==========================================================
# STEP 5 - FEATURE IMPORTANCE
# ==========================================================

print("\n" + "="*60)
print("STEP 5 - FEATURE IMPORTANCE")
print("="*60)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

# ==========================================================
# STEP 6 - TOP 10 FEATURES
# ==========================================================

print("\n" + "="*60)
print("TOP 10 IMPORTANT FEATURES")
print("="*60)

top10 = feature_importance.head(10)

print(top10)

# ==========================================================
# STEP 7 - GRAPH
# ==========================================================

plt.figure(figsize=(10,6))

plt.barh(top10["Feature"], top10["Importance"])

plt.title("Top 10 Important Features")

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.gca().invert_yaxis()

plt.tight_layout()

# Save the graph
plt.savefig("reports/feature_importance.png", dpi=300)

# Display the graph
plt.show()

print("\nFeature Importance Graph Displayed Successfully!")
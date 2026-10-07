# ==========================================
# RANDOM FOREST CLASSIFIER
# ==========================================

from xml.parsers.expat import model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ==========================================
# STEP 1 - LOAD DATASET
# ==========================================

print("=" * 60)
print("STEP 1 - LOADING DATASET")
print("=" * 60)

df = pd.read_csv("dataset/student_dataset_feature_engineered.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ==========================================
# STEP 2 - ENCODE TARGET COLUMN
# ==========================================

print("\n" + "=" * 60)
print("STEP 2 - ENCODING TARGET COLUMN")
print("=" * 60)

label_encoder = LabelEncoder()

df["Placement_Status"] = label_encoder.fit_transform(df["Placement_Status"])

print(df["Placement_Status"].head())

# ==========================================
# STEP 3 - FEATURES AND TARGET
# ==========================================

print("\n" + "=" * 60)
print("STEP 3 - FEATURES AND TARGET")
print("=" * 60)

X = df.drop(["StudentID", "Placement_Status"], axis=1)

y = df["Placement_Status"]

print("Features Shape :", X.shape)
print("Target Shape :", y.shape)

# ==========================================
# STEP 4 - TRAIN TEST SPLIT
# ==========================================

print("\n" + "=" * 60)
print("STEP 4 - TRAIN TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data :", X_train.shape)
print("Testing Data :", X_test.shape)

# ==========================================
# STEP 5 - CREATE RANDOM FOREST MODEL
# ==========================================

print("\n" + "=" * 60)
print("STEP 5 - RANDOM FOREST MODEL")
print("=" * 60)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)

print("SVM Model Created Successfully!")

# ==========================================
# STEP 6 - TRAIN MODEL
# ==========================================

print("\n" + "=" * 60)
print("STEP 6 - TRAINING MODEL")
print("=" * 60)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================================
# STEP 7 - PREDICTION
# ==========================================

print("\n" + "=" * 60)
print("STEP 7 - PREDICTION")
print("=" * 60)

y_pred = model.predict(X_test)

print("Prediction Completed!")

# ==========================================
# STEP 8 - MODEL EVALUATION
# ==========================================

print("\n" + "=" * 60)
print("STEP 8 - MODEL EVALUATION")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ==========================================
# STEP 9 - CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(y_test, y_pred))

# ==========================================
# STEP 10 - CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================
# STEP 11 - ROC CURVE
# ==========================================

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, linewidth=3, label=f"AUC = {auc:.3f}")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.legend()
plt.grid(True)

plt.show()

# ==========================================
# STEP 12 - SAVE MODEL
# ==========================================

joblib.dump(model, "models/xgboost.pkl")

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("Model saved as: models/xgboost.pkl")
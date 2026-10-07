import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("dataset/student_dataset_feature_engineered.csv")

# --------------------------------------------------
# Remove StudentID
# --------------------------------------------------

X = df.drop(["StudentID", "Placement_Status"], axis=1)

y = df["Placement_Status"]

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Create Logistic Regression Model
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)

# --------------------------------------------------
# Train the Model
# --------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print("Accuracy :", accuracy_score(y_test, y_pred))

print("Precision :", precision_score(y_test, y_pred))

print("Recall :", recall_score(y_test, y_pred))

print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, y_pred))
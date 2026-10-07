import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("dataset/student_dataset_feature_engineered.csv")

# -----------------------------------
# Remove StudentID (Not useful for prediction)
# -----------------------------------
df = df.drop("StudentID", axis=1)

# -----------------------------------
# Split Features and Target
# -----------------------------------
X = df.drop("Placement_Status", axis=1)
y = df["Placement_Status"]

print("=" * 60)
print("FEATURES SHAPE")
print(X.shape)

print("\nTARGET SHAPE")
print(y.shape)

# -----------------------------------
# Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAINING DATA")
print("=" * 60)
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("\n" + "=" * 60)
print("TESTING DATA")
print("=" * 60)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)
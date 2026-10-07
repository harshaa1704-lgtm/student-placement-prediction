import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("dataset/student_dataset_cleaned.csv")

print("=" * 60)
print("DATASET BEFORE FEATURE ENGINEERING")
print("=" * 60)
print(df.head())

# ---------------------------------------
# Encode Categorical Columns
# ---------------------------------------

encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "College_Tier",
    "Specialization"
]

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

print("\n")
print("=" * 60)
print("DATASET AFTER FEATURE ENGINEERING")
print("=" * 60)
print(df.head())

# ---------------------------------------
# Separate Features and Target
# ---------------------------------------

X = df.drop("Placement_Status", axis=1)

y = df["Placement_Status"]

print("\n")
print("=" * 60)
print("FEATURES (X)")
print("=" * 60)
print(X.head())

print("\n")
print("=" * 60)
print("TARGET (y)")
print("=" * 60)
print(y.head())

# ---------------------------------------
# Save Dataset
# ---------------------------------------

df.to_csv(
    "dataset/student_dataset_feature_engineered.csv",
    index=False
)

print("\n")
print("=" * 60)
print("Feature Engineering Completed Successfully")
print("Dataset Saved Successfully")
print("=" * 60)
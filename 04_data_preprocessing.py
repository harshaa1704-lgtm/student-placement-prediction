import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("dataset/student_dataset_COMPLETE_10000.csv")

print("========== BEFORE ENCODING ==========")
print(df.head())

# Create Label Encoder
encoder = LabelEncoder()

# Encode categorical columns
df["Gender"] = encoder.fit_transform(df["Gender"])
df["College_Tier"] = encoder.fit_transform(df["College_Tier"])
df["Specialization"] = encoder.fit_transform(df["Specialization"])
df["Placement_Status"] = encoder.fit_transform(df["Placement_Status"])

print("\n========== AFTER ENCODING ==========")
print(df.head())

# Save the cleaned dataset
df.to_csv("dataset/student_dataset_cleaned.csv", index=False)

print("\n✅ Cleaned dataset saved successfully!")
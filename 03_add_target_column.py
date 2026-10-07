import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("dataset/student_dataset_COMPLETE_10000.csv")

# Create Placement_Status based on student performance
conditions = (
    (df["CGPA"] >= 7.0) &
    (df["Coding_Skill_Score"] >= 60) &
    (df["Communication_Skills"] >= 7.0) &
    (df["Internships"] >= 1)
)

# Assign Placed or Not Placed
df["Placement_Status"] = np.where(conditions, "Placed", "Not Placed")

# Save the updated dataset
df.to_csv("dataset/student_dataset_COMPLETE_10000.csv", index=False)

print("✅ Placement_Status column added successfully!")
print(df[["CGPA", "Coding_Skill_Score", "Communication_Skills", "Internships", "Placement_Status"]].head())
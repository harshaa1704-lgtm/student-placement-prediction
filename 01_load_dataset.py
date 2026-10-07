import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/student_dataset_COMPLETE_10000.csv")

# Display first 5 rows
print("========== FIRST 5 ROWS ==========")
print(df.head())
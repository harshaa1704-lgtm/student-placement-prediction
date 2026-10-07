import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# Load Dataset
# =====================================
dataset_path = r"D:\Mini project\Student_Placement_Project\dataset\student_dataset_feature_engineered.csv"

df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully")
print(df.head())

# =====================================
# Check Missing Values
# =====================================
print("\nMissing Values:")
print(df.isnull().sum())

# =====================================
# Remove rows with missing target
# =====================================
df = df.dropna(subset=["Placement_Status"])

# =====================================
# Convert target column if needed
# =====================================
if df["Placement_Status"].dtype == object:
    df["Placement_Status"] = (
        df["Placement_Status"]
        .astype(str)
        .str.strip()
        .replace({
            "Placed": 1,
            "Not Placed": 0,
            "Yes": 1,
            "No": 0
        })
    )

df = df.dropna(subset=["Placement_Status"])
df["Placement_Status"] = df["Placement_Status"].astype(int)

print("\nTarget Value Counts:")
print(df["Placement_Status"].value_counts())

# =====================================
# Drop unnecessary columns
# =====================================
if "StudentID" in df.columns:
    df.drop(columns=["StudentID"], inplace=True)

# =====================================
# Separate Features and Target
# =====================================
X = df.drop("Placement_Status", axis=1)
y = df["Placement_Status"]

# =====================================
# Convert categorical columns
# =====================================
X = pd.get_dummies(X, drop_first=True)
joblib.dump(
    X.columns.tolist(),
    r"D:\Mini project\Student_Placement_Project\models\feature_names.pkl"
)

# =====================================
# Check for NaN values
# =====================================
print("\nNaN values in X:", X.isna().sum().sum())
print("NaN values in y:", y.isna().sum())

# Remove rows with NaN in features (if any)
X = X.fillna(0)

# =====================================
# Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Feature Scaling
# =====================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# Train Model
# =====================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =====================================
# Prediction
# =====================================
y_pred = model.predict(X_test_scaled)

# =====================================
# Evaluation
# =====================================
accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print(f"Accuracy : {accuracy:.4f}")
print("====================================")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# =====================================
# Save Model & Scaler
# =====================================
joblib.dump(model, r"D:\Mini project\Student_Placement_Project\models\best_model.pkl")
joblib.dump(scaler, r"D:\Mini project\Student_Placement_Project\models\scaler.pkl")

print("\nModel Saved Successfully!")
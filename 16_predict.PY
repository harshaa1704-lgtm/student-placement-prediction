import pandas as pd
import joblib

# ==========================================
# Load Trained Model, Scaler & Feature Names
# ==========================================
model = joblib.load(r"D:\Mini project\Student_Placement_Project\models\best_model.pkl")
scaler = joblib.load(r"D:\Mini project\Student_Placement_Project\models\scaler.pkl")
feature_names = joblib.load(r"D:\Mini project\Student_Placement_Project\models\feature_names.pkl")

# ==========================================
# Enter Student Details
# ==========================================

student = {
    "Gender": "Male",
    "College_Tier": 2,
    "Specialization": "CSE",
    "CGPA": 8.5,
    "SSC_Marks": 90,
    "HSC_Marks": 88,
    "DSA_Problems_Solved": 350,
    "Internships": 2,
    "Certifications": 5,
    "Projects_Count": 4,
    "Communication_Skills": 8,
    "Aptitude_Test_Score": 85,
    "LeetCode_Rating": 1700,
    "GitHub_Contributions": 120,
    "Hackathons_Participated": 3,
    "AI_ML_Skill_Level": 8,
    "System_Design_Knowledge": 7,
    "Resume_Score": 85,
    "Mock_Interview_Score": 82,
    "SoftSkillsRating": 8,
    "Study_Hours_Per_Day": 5,
    "Coding_Skill_Score": 90
}

# ==========================================
# Convert to DataFrame
# ==========================================
input_df = pd.DataFrame([student])

# ==========================================
# One-Hot Encoding
# ==========================================
input_df = pd.get_dummies(input_df)

# Add missing columns
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# ==========================================
# Scale Features
# ==========================================
input_scaled = scaler.transform(input_df)

# ==========================================
# Prediction
# ==========================================
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

# ==========================================
# Display Result
# ==========================================
print("\n========== Prediction Result ==========")

if prediction == 1:
    print("Placement Prediction : PLACED")
else:
    print("Placement Prediction : NOT PLACED")

print(f"Probability (Not Placed): {probability[0]*100:.2f}%")
print(f"Probability (Placed)    : {probability[1]*100:.2f}%")

print("=======================================")
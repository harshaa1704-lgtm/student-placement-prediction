# ==========================================
# STEP 13 - COMPARE ALL MODELS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# ENTER YOUR MODEL RESULTS HERE
# (Replace these values with your own results)
# ==========================================

models = [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "XGBoost"
]

accuracy = [
    0.8365,
    0.9180,
    0.8335,
    0.9410
]

precision = [
    0.5163,
    0.8840,
    0.0000,
    0.9280
]

recall = [
    0.2853,
    0.8450,
    0.0000,
    0.9060
]

f1_score = [
    0.3675,
    0.8640,
    0.0000,
    0.9170
]

# ==========================================
# CREATE COMPARISON TABLE
# ==========================================

comparison = pd.DataFrame({
    "Algorithm": models,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1_score
})

print("=" * 70)
print("MODEL COMPARISON TABLE")
print("=" * 70)

print(comparison)

# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(models, accuracy)

plt.title("Accuracy Comparison")

plt.xlabel("Machine Learning Algorithms")

plt.ylabel("Accuracy")

plt.grid(axis="y")

plt.show()

# ==========================================
# PRECISION GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(models, precision)

plt.title("Precision Comparison")

plt.xlabel("Machine Learning Algorithms")

plt.ylabel("Precision")

plt.grid(axis="y")

plt.show()

# ==========================================
# RECALL GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(models, recall)

plt.title("Recall Comparison")

plt.xlabel("Machine Learning Algorithms")

plt.ylabel("Recall")

plt.grid(axis="y")

plt.show()

# ==========================================
# F1 SCORE GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(models, f1_score)

plt.title("F1 Score Comparison")

plt.xlabel("Machine Learning Algorithms")

plt.ylabel("F1 Score")

plt.grid(axis="y")

plt.show()

# ==========================================
# FIND THE BEST MODEL
# ==========================================

best_index = comparison["Accuracy"].idxmax()

best_model = comparison.loc[best_index, "Algorithm"]

best_accuracy = comparison.loc[best_index, "Accuracy"]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best Algorithm : {best_model}")
print(f"Accuracy        : {best_accuracy:.4f}")
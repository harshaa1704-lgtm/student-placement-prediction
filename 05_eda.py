import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/student_dataset_cleaned.csv")

# Professional Style
sns.set_style("whitegrid")

# =========================================================
# GRAPH 1 : Placement Status Distribution
# =========================================================
plt.figure(figsize=(8,6))

ax = sns.countplot(
    x="Placement_Status",
    data=df,
    hue="Placement_Status",
    palette="Set2",
    legend=False
)

plt.title("Distribution of Student Placement Status",
          fontsize=18,
          fontweight="bold")

plt.xlabel("Placement Status", fontsize=14)
plt.ylabel("Number of Students", fontsize=14)

for container in ax.containers:
    ax.bar_label(container, fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 2 : CGPA Distribution
# =========================================================
plt.figure(figsize=(8,6))

sns.histplot(
    df["CGPA"],
    bins=15,
    kde=True,
    color="royalblue"
)

plt.title("Distribution of Student CGPA",
          fontsize=18,
          fontweight="bold")

plt.xlabel("CGPA", fontsize=14)
plt.ylabel("Number of Students", fontsize=14)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 3 : College Tier Distribution
# =========================================================
plt.figure(figsize=(8,6))

ax = sns.countplot(
    x="College_Tier",
    data=df,
    hue="College_Tier",
    palette="viridis",
    legend=False
)

plt.title("Distribution of Students by College Tier",
          fontsize=18,
          fontweight="bold")

plt.xlabel("College Tier", fontsize=14)
plt.ylabel("Number of Students", fontsize=14)

for container in ax.containers:
    ax.bar_label(container)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 4 : Specialization Distribution
# =========================================================
plt.figure(figsize=(12,6))

ax = sns.countplot(
    x="Specialization",
    data=df,
    hue="Specialization",
    palette="tab10",
    legend=False
)

plt.title("Distribution of Students by Specialization",
          fontsize=18,
          fontweight="bold")

plt.xlabel("Specialization", fontsize=14)
plt.ylabel("Number of Students", fontsize=14)

plt.xticks(rotation=45)

for container in ax.containers:
    ax.bar_label(container)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 5 : Correlation Heatmap
# =========================================================
plt.figure(figsize=(16,12))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features",
          fontsize=18,
          fontweight="bold")

plt.tight_layout()
plt.show()

print("\n===================================")
print(" All EDA Graphs Generated Successfully")
print("===================================")
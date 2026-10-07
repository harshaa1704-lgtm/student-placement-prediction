# ==========================================
# STEP 14 - HYPERPARAMETER TUNING (XGBoost)
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

# ==========================================
# LOAD DATASET
# ==========================================

print("="*60)
print("LOADING DATASET")
print("="*60)

df = pd.read_csv("dataset/student_dataset_feature_engineered.csv")

print(df.head())

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(["StudentID","Placement_Status"], axis=1)

X = pd.get_dummies(X)

y = df["Placement_Status"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# PARAMETER GRID
# ==========================================

parameters = {

    "n_estimators":[50,100,150],

    "max_depth":[3,5,7],

    "learning_rate":[0.01,0.1,0.2]

}

# ==========================================
# XGBOOST MODEL
# ==========================================

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

# ==========================================
# GRID SEARCH
# ==========================================

print("="*60)
print("STARTING GRID SEARCH")
print("="*60)

grid = GridSearchCV(

    estimator=xgb,

    param_grid=parameters,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(X_train,y_train)

# ==========================================
# BEST PARAMETERS
# ==========================================

print("="*60)
print("BEST PARAMETERS")
print("="*60)

print(grid.best_params_)

print()

print("Best Cross Validation Accuracy :",grid.best_score_)

# ==========================================
# BEST MODEL
# ==========================================

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("="*60)
print("TEST ACCURACY")
print("="*60)

print(accuracy_score(y_test,y_pred))

print()

print(classification_report(y_test,y_pred))
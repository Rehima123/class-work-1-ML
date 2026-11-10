import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ============================================================
# PART 1. CREATE DATASET (30 RECORDS - Normal Focus)
# Target: 1 = Normal, 0 = Other (Warning or Fault)
# ============================================================

data = [
    # ID, Vibration_mm_s, RPM, Temperature_C, Pressure_bar, Operating_Hours, Is_Normal
    [1,  1.1, 1500,  45, 10.2,  800, 1],  # Normal
    [2,  1.4, 1490,  50, 10.0, 1100, 1],  # Normal
    [3,  1.7, 1480,  54,  9.8, 1400, 1],  # Normal
    [4,  1.9, 1495,  57,  9.5, 1700, 1],  # Normal
    [5,  2.2, 1460,  61,  9.1, 2100, 1],  # Normal
    [6,  1.3, 1505,  52, 10.1, 1050, 1],  # Normal
    [7,  2.0, 1455,  59,  9.3, 2150, 1],  # Normal
    [8,  1.6, 1490,  56,  9.7, 1550, 1],  # Normal
    [9,  2.5, 1435,  64,  8.9, 2600, 1],  # Normal
    [10, 1.2, 1510,  49, 10.3,  950, 1],  # Normal

    [11, 3.5, 1410,  70,  7.8, 3200, 0],  # Warning (Not Normal)
    [12, 3.8, 1395,  73,  7.5, 3600, 0],  # Warning
    [13, 4.2, 1380,  76,  7.1, 4100, 0],  # Warning
    [14, 3.6, 1400,  72,  7.6, 3400, 0],  # Warning
    [15, 4.5, 1365,  79,  6.8, 4500, 0],  # Warning
    [16, 3.9, 1390,  74,  7.3, 3800, 0],  # Warning
    [17, 4.7, 1350,  81,  6.5, 4800, 0],  # Warning
    [18, 3.4, 1405,  69,  7.9, 3100, 0],  # Warning
    [19, 4.1, 1385,  75,  7.2, 3900, 0],  # Warning
    [20, 4.8, 1340,  83,  6.2, 5000, 0],  # Warning

    [21, 6.0, 1300,  86,  5.5, 5600, 0],  # Fault (Not Normal)
    [22, 6.5, 1280,  89,  5.1, 6100, 0],  # Fault
    [23, 7.1, 1250,  92,  4.6, 6800, 0],  # Fault
    [24, 5.8, 1310,  85,  5.7, 5400, 0],  # Fault
    [25, 7.6, 1220,  95,  4.1, 7400, 0],  # Fault
    [26, 6.3, 1290,  88,  5.3, 5900, 0],  # Fault
    [27, 8.2, 1200,  98,  3.6, 8100, 0],  # Fault
    [28, 5.5, 1320,  84,  5.9, 5200, 0],  # Fault
    [29, 6.9, 1260,  91,  4.8, 6600, 0],  # Fault
    [30, 8.5, 1180, 101,  3.2, 8600, 0],  # Fault
]

columns = [
    "Pump_ID", "Vibration_mm_s", "RPM", "Temperature_C",
    "Pressure_bar", "Operating_Hours", "Is_Normal"
]
df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 PUMP RECORDS (Normal Focus)")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Is_Normal"].value_counts().rename(index={1: "Normal", 0: "Not Normal (Warning/Fault)"}))

# Separate predictors (X) and target (y).
feature_cols = ["Vibration_mm_s", "RPM", "Temperature_C", "Pressure_bar", "Operating_Hours"]
X = df[feature_cols]
y = df["Is_Normal"]

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ============================================================
# PART 3. DEFINE FIVE CLASSIFIERS
# ============================================================

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

decision_tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50, learning_rate=0.5, random_state=42
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE
# ============================================================

results = []

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print(f"{model_name.upper()} - NORMAL STAGE DETECTION")
    print("=" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Not Normal (Warning/Fault)", "Normal"],
        zero_division=0
    ))

# ============================================================
# PART 5. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON (NORMAL STAGE FOCUS)")
print(results_df.round(3))

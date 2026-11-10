"""
Practical Engineering Case Study: Pump Condition Classification
3.12 Classwork — Classification Based Supervised Learning

Problem1: classify each pump observation as Normal, Warning, or Fault
          using labelled historical data.

Input features    : Vibration (mm/s), RPM, Temperature (°C),
                    Pressure (bar), Operating Hours
Target label      : Normal / Warning / Fault
Possible source   : Sensor logs and maintenance inspection records
Algorithms        : Decision Tree, Random Forest, SVM, Naïve Bayes, AdaBoost
Evaluation        : Recall for Fault, Precision for Fault, Macro F1,
                    Confusion Matrix
Caution           : Predictions should support—not replace—engineering
                    inspection and safety procedures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ============================================================
# PART 1. CREATE DATASET (30 RECORDS)
# Target label: 0 = Normal | 1 = Warning | 2 = Fault
# ============================================================

data = [
    # ID, Vibration_mm_s, RPM, Temperature_C, Pressure_bar, Operating_Hours, Condition
    # --- NORMAL (10 records) ---
    [1,  1.1, 1500,  45, 10.2,  800, 0],
    [2,  1.4, 1490,  50, 10.0, 1100, 0],
    [3,  1.7, 1480,  54,  9.8, 1400, 0],
    [4,  1.9, 1495,  57,  9.5, 1700, 0],
    [5,  2.2, 1460,  61,  9.1, 2100, 0],
    [6,  1.3, 1505,  52, 10.1, 1050, 0],
    [7,  2.0, 1455,  59,  9.3, 2150, 0],
    [8,  1.6, 1490,  56,  9.7, 1550, 0],
    [9,  2.5, 1435,  64,  8.9, 2600, 0],
    [10, 1.2, 1510,  49, 10.3,  950, 0],
    # --- WARNING (10 records) ---
    [11, 3.5, 1410,  70,  7.8, 3200, 1],
    [12, 3.8, 1395,  73,  7.5, 3600, 1],
    [13, 4.2, 1380,  76,  7.1, 4100, 1],
    [14, 3.6, 1400,  72,  7.6, 3400, 1],
    [15, 4.5, 1365,  79,  6.8, 4500, 1],
    [16, 3.9, 1390,  74,  7.3, 3800, 1],
    [17, 4.7, 1350,  81,  6.5, 4800, 1],
    [18, 3.4, 1405,  69,  7.9, 3100, 1],
    [19, 4.1, 1385,  75,  7.2, 3900, 1],
    [20, 4.8, 1340,  83,  6.2, 5000, 1],
    # --- FAULT (10 records) ---
    [21, 6.0, 1300,  86,  5.5, 5600, 2],
    [22, 6.5, 1280,  89,  5.1, 6100, 2],
    [23, 7.1, 1250,  92,  4.6, 6800, 2],
    [24, 5.8, 1310,  85,  5.7, 5400, 2],
    [25, 7.6, 1220,  95,  4.1, 7400, 2],
    [26, 6.3, 1290,  88,  5.3, 5900, 2],
    [27, 8.2, 1200,  98,  3.6, 8100, 2],
    [28, 5.5, 1320,  84,  5.9, 5200, 2],
    [29, 6.9, 1260,  91,  4.8, 6600, 2],
    [30, 8.5, 1180, 101,  3.2, 8600, 2],
]

columns = [
    "Pump_ID", "Vibration_mm_s", "RPM", "Temperature_C",
    "Pressure_bar", "Operating_Hours", "Condition"
]
df = pd.DataFrame(data, columns=columns)

# Human-readable target label column
label_map = {0: "Normal", 1: "Warning", 2: "Fault"}
df["Condition_Label"] = df["Condition"].map(label_map)

CLASS_NAMES = ["Normal", "Warning", "Fault"]

print("\nDATASET: 30 PUMP SENSOR RECORDS")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Condition_Label"].value_counts())

# Separate predictors (X) and target (y)
feature_cols = ["Vibration_mm_s", "RPM", "Temperature_C", "Pressure_bar", "Operating_Hours"]
X = df[feature_cols]
y = df["Condition"]

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")

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
random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42
)
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50, learning_rate=0.5, random_state=42
)

models = {
    "SVM":           svm_model,
    "Naive Bayes":   naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost":      adaboost_model
}

# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE
# ============================================================

results = []

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print(f"  {model_name.upper()} - PUMP CONDITION DETECTION")
    print("=" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy     = accuracy_score(y_test, y_pred)
    macro_f1     = f1_score(y_test, y_pred, average="macro",  zero_division=0)
    # Fault-specific metrics — evaluation priority per classwork spec
    fault_prec   = precision_score(y_test, y_pred, labels=[2], average="micro", zero_division=0)
    fault_recall = recall_score(y_test, y_pred,    labels=[2], average="micro", zero_division=0)

    results.append({
        "Algorithm":       model_name,
        "Accuracy":        accuracy,
        "Macro F1":        macro_f1,
        "Fault Precision": fault_prec,
        "Fault Recall":    fault_recall,
    })

    print(f"Accuracy        : {accuracy:.3f}")
    print(f"Macro F1        : {macro_f1:.3f}")
    print(f"Fault Precision : {fault_prec:.3f}  ← evaluation priority")
    print(f"Fault Recall    : {fault_recall:.3f}  ← evaluation priority")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 5. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON (Normal / Warning / Fault)")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Pump Condition Classification — Algorithm Comparison")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

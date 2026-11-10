
"""
Practical Engineering Case Study:
Pump Fault Classification (Binary)

Target:
1 = Fault
0 = Not Fault (Normal or Warning)
"""

# ============================================================
# PART 1. IMPORT LIBRARIES
# ============================================================

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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# PART 2. CREATE DATASET (30 RECORDS - FAULT FOCUS)
# Target: 1 = Fault, 0 = Not Fault
# ============================================================

data = [
    # ID, Vibration, RPM, Temperature, Pressure, Hours, Is_Fault

    # NORMAL (Not Fault = 0)
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

    # WARNING (Not Fault = 0)
    [11, 3.5, 1410,  70,  7.8, 3200, 0],
    [12, 3.8, 1395,  73,  7.5, 3600, 0],
    [13, 4.2, 1380,  76,  7.1, 4100, 0],
    [14, 3.6, 1400,  72,  7.6, 3400, 0],
    [15, 4.5, 1365,  79,  6.8, 4500, 0],
    [16, 3.9, 1390,  74,  7.3, 3800, 0],
    [17, 4.7, 1350,  81,  6.5, 4800, 0],
    [18, 3.4, 1405,  69,  7.9, 3100, 0],
    [19, 4.1, 1385,  75,  7.2, 3900, 0],
    [20, 4.8, 1340,  83,  6.2, 5000, 0],

    # FAULT (Fault = 1)
    [21, 6.0, 1300,  86,  5.5, 5600, 1],
    [22, 6.5, 1280,  89,  5.1, 6100, 1],
    [23, 7.1, 1250,  92,  4.6, 6800, 1],
    [24, 5.8, 1310,  85,  5.7, 5400, 1],
    [25, 7.6, 1220,  95,  4.1, 7400, 1],
    [26, 6.3, 1290,  88,  5.3, 5900, 1],
    [27, 8.2, 1200,  98,  3.6, 8100, 1],
    [28, 5.5, 1320,  84,  5.9, 5200, 1],
    [29, 6.9, 1260,  91,  4.8, 6600, 1],
    [30, 8.5, 1180, 101,  3.2, 8600, 1],
]

columns = [
    "Pump_ID",
    "Vibration_mm_s",
    "RPM",
    "Temperature_C",
    "Pressure_bar",
    "Operating_Hours",
    "Is_Fault"
]

df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 PUMP RECORDS (FAULT FOCUS)")
print(df.to_string(index=False))

print("\nCLASS DISTRIBUTION:")
print(
    df["Is_Fault"].value_counts()
    .rename(index={0: "Not Fault", 1: "Fault"})
)

# ============================================================
# PART 3. SEPARATE INPUT FEATURES (X) AND TARGET (y)
# ============================================================

feature_cols = [
    "Vibration_mm_s",
    "RPM",
    "Temperature_C",
    "Pressure_bar",
    "Operating_Hours"
]

X = df[feature_cols]
y = df["Is_Fault"]

print("\nINPUT FEATURES:")
print(X.head())

print("\nTARGET LABEL:")
print(y.head())

# ============================================================
# PART 4. TRAIN-TEST SPLIT (70% TRAINING, 30% TESTING)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA SIZE:", len(X_train))
print("TESTING DATA SIZE:", len(X_test))

# ============================================================
# PART 5. DEFINE FIVE CLASSIFICATION ALGORITHMS
# ============================================================

# 1. SUPPORT VECTOR MACHINE (SVM)

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ))
])

# 2. NAIVE BAYES

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

# 3. DECISION TREE

decision_tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# 4. RANDOM FOREST

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

# 5. ADABOOST

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

# Store all five models

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# ============================================================
# PART 6. TRAIN, PREDICT AND EVALUATE
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(f"{model_name.upper()} - FAULT DETECTION")
    print("=" * 65)

    # Train the model
    model.fit(X_train, y_train)

    # Predict testing data
    y_pred = model.predict(X_test)

    predictions[model_name] = y_pred

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # Store results
    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Display results
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    # Classification report
    print("\nCLASSIFICATION REPORT:")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Fault", "Fault"],
        zero_division=0
    ))

# ============================================================
# PART 7. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")

print("\n" + "=" * 65)
print("MODEL COMPARISON (FAULT CLASSIFICATION)")
print("=" * 65)

print(results_df.round(3))

# Find the model with the highest Fault recall
best_model = results_df["Recall"].idxmax()

print("\nBEST MODEL BASED ON FAULT RECALL:")
print(best_model)

# ============================================================
# PART 8. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes = axes.flatten()

for i, (model_name, y_pred) in enumerate(predictions.items()):

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fault", "Fault"],
        yticklabels=["Not Fault", "Fault"],
        ax=axes[i]
    )

    axes[i].set_title(f"{model_name} Confusion Matrix")
    axes[i].set_xlabel("Predicted Condition")
    axes[i].set_ylabel("Actual Condition")

axes[-1].axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# PART 9. VISUALIZE MODEL PERFORMANCE
# ============================================================

results_df.plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1.1)
)

plt.title("Machine Learning Model Comparison - Fault Detection")
plt.xlabel("Classification Algorithms")
plt.ylabel("Performance Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# PART 10. DECISION TREE VISUALIZATION
# ============================================================

plt.figure(figsize=(16, 8))

plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Not Fault", "Fault"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Pump Fault Classification")
plt.show()

# ============================================================
# PART 11. TEST A NEW PUMP OBSERVATION
# ============================================================

new_pump = pd.DataFrame([{
    "Vibration_mm_s": 7.0,
    "RPM": 1250,
    "Temperature_C": 93,
    "Pressure_bar": 4.5,
    "Operating_Hours": 7000
}])

print("\nNEW PUMP OBSERVATION:")
print(new_pump)

print("\nPREDICTIONS FROM ALL FIVE MODELS:")

for model_name, model in models.items():

    prediction = model.predict(new_pump)[0]

    condition = "FAULT" if prediction == 1 else "NOT FAULT"

    print(f"{model_name}: {condition}")

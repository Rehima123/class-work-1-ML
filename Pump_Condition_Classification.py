"""
Practical Engineering Case Study: Pump Warning Detection (Binary Classification)
Target: Classify if a pump observation is a Warning (1) or Not Warning (0).
Algorithms: SVM, Naïve Bayes, Decision Tree, Random Forest, AdaBoost
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
# PART 1. CREATE DATASET (30 RECORDS - Warning Focus)
# Target: 1 = Warning, 0 = Other (Normal or Fault)
# ============================================================

data = [
    # ID, Vibration_mm_s, RPM, Temperature_C, Pressure_bar, Operating_Hours, Is_Warning
    [1,  1.1, 1500,  45, 10.2,  800, 0],  # Normal (Not Warning)
    [2,  1.4, 1490,  50, 10.0, 1100, 0],  # Normal
    [3,  1.7, 1480,  54,  9.8, 1400, 0],  # Normal
    [4,  1.9, 1495,  57,  9.5, 1700, 0],  # Normal
    [5,  2.2, 1460,  61,  9.1, 2100, 0],  # Normal
    [6,  1.3, 1505,  52, 10.1, 1050, 0],  # Normal
    [7,  2.0, 1455,  59,  9.3, 2150, 0],  # Normal
    [8,  1.6, 1490,  56,  9.7, 1550, 0],  # Normal
    [9,  2.5, 1435,  64,  8.9, 2600, 0],  # Normal
    [10, 1.2, 1510,  49, 10.3,  950, 0],  # Normal
    
    [11, 3.5, 1410,  70,  7.8, 3200, 1],  # Warning (Target = 1)
    [12, 3.8, 1395,  73,  7.5, 3600, 1],  # Warning
    [13, 4.2, 1380,  76,  7.1, 4100, 1],  # Warning
    [14, 3.6, 1400,  72,  7.6, 3400, 1],  # Warning
    [15, 4.5, 1365,  79,  6.8, 4500, 1],  # Warning
    [16, 3.9, 1390,  74,  7.3, 3800, 1],  # Warning
    [17, 4.7, 1350,  81,  6.5, 4800, 1],  # Warning
    [18, 3.4, 1405,  69,  7.9, 3100, 1],  # Warning
    [19, 4.1, 1385,  75,  7.2, 3900, 1],  # Warning
    [20, 4.8, 1340,  83,  6.2, 5000, 1],  # Warning
    
    [21, 6.0, 1300,  86,  5.5, 5600, 0],  # Fault (Severe - Not Warning)
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
    "Pressure_bar", "Operating_Hours", "Is_Warning"
]
df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 PUMP RECORDS (Warning Focus)")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Is_Warning"].value_counts().rename(index={0: "Other (Normal/Fault)", 1: "Warning"}))

# --- Scatter plot: Vibration vs Temperature ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Vibration_mm_s", y="Temperature_C",
    hue="Is_Warning", style="Is_Warning",
    palette={0: "steelblue", 1: "tomato"}, s=100
)
plt.title("Pump: Vibration vs Temperature by Class")
plt.xlabel("Vibration (mm/s)")
plt.ylabel("Temperature (°C)")
plt.legend(title="0=Other, 1=Warning")
plt.tight_layout()
plt.show()

# --- Correlation heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(
    df[["Vibration_mm_s", "RPM", "Temperature_C", "Pressure_bar", "Operating_Hours", "Is_Warning"]].corr(),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Matrix – Pump Features")
plt.tight_layout()
plt.show()

# Separate predictors (X) and target (y).
feature_cols = ["Vibration_mm_s", "RPM", "Temperature_C", "Pressure_bar", "Operating_Hours"]
X = df[feature_cols]
y = df["Is_Warning"]

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

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
# PART 4. TRAIN, PREDICT, AND EVALUATE EACH MODEL
# ============================================================

results = []

for model_name, model in models.items() :
    print("\n" + "=" * 65)
    print(f"{model_name.upper()} - WARNING DETECTION TRAINING")
    print("=" * 65)

    # Train model
    model.fit(X_train, y_train)

    # Predict test set
    y_pred = model.predict(X_test)

    # Metrics
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
        target_names=["Other", "Warning"],
        zero_division=0
    ))

    # Confusion Matrix Visualization
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Other", "Warning"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Warning Detection CM")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 5. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON (WARNING DETECTION)")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(10, 5), ylim=(0, 1.05))
plt.title("Algorithm Comparison for Pump Warning Detection")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 6. FEATURE IMPORTANCE (TREE-BASED MODELS)
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values, color="steelblue")
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# Decision Tree visualization
plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Other", "Warning"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree – Pump Warning Detection")
plt.tight_layout()
plt.show()

# ============================================================
# PART 7. TEST A NEW PUMP OBSERVATION FOR WARNING
# ============================================================

new_observation = pd.DataFrame([{
    "Vibration_mm_s": 3.7,
    "RPM": 1390,
    "Temperature_C": 72,
    "Pressure_bar": 7.4,
    "Operating_Hours": 3500
}])

print("\nNEW PUMP OBSERVATION TEST")
print(new_observation)

for name, model in models.items():
    pred = int(model.predict(new_observation)[0])
    label = "Warning" if pred == 1 else "Other (Normal/Fault)"
    print(f"{name:15s} prediction: {label}")

print("\nEND OF WARNING CLASSIFICATION CASE STUDY")
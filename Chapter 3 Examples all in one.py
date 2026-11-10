"""
Practical Engineering Case Study: Supervised Machine Learning Classification
Domain: Civil/Structural Engineering – Bridge Structural Health Monitoring
Algorithms:
1. Support Vector Machines (SVM)
2. Naïve Bayes (Gaussian Naïve Bayes)
3. Decision Tree
4. Random Forest
5. AdaBoost
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
# Domain: Bridge Structural Health Monitoring
# Features:
#   Load_kN          – Live traffic load on the bridge deck (kN)
#   Deflection_mm    – Midspan vertical deflection (mm)
#   Crack_Width_mm   – Observed surface crack width (mm)
#   Age_Years        – Age of the bridge structure (years)
# Target:
#   Structural_Risk  – 0 = Structurally sound, 1 = At-risk / needs intervention
# ============================================================

data = [
    # ID, Load_kN, Deflection_mm, Crack_Width_mm, Age_Years, Structural_Risk
    [1,  120,  2.1, 0.05, 10, 0],
    [2,  135,  2.4, 0.08, 12, 0],
    [3,  110,  1.9, 0.04, 8,  0],
    [4,  145,  2.7, 0.09, 15, 0],
    [5,  130,  2.3, 0.06, 11, 0],
    [6,  125,  2.2, 0.07, 9,  0],
    [7,  140,  2.6, 0.08, 13, 0],
    [8,  115,  2.0, 0.05, 7,  0],
    [9,  150,  2.9, 0.10, 16, 0],
    [10, 128,  2.3, 0.06, 10, 0],
    [11, 138,  2.5, 0.07, 14, 0],
    [12, 122,  2.1, 0.05, 9,  0],
    [13, 142,  2.7, 0.09, 15, 0],
    [14, 118,  2.0, 0.05, 8,  0],
    [15, 132,  2.4, 0.07, 12, 0],
    [16, 310,  8.5, 0.55, 42, 1],
    [17, 345,  9.8, 0.72, 50, 1],
    [18, 290,  7.9, 0.48, 38, 1],
    [19, 370, 10.4, 0.85, 55, 1],
    [20, 325,  9.1, 0.63, 47, 1],
    [21, 360, 10.0, 0.80, 53, 1],
    [22, 280,  7.6, 0.44, 36, 1],
    [23, 395, 11.2, 0.92, 60, 1],
    [24, 305,  8.3, 0.52, 41, 1],
    [25, 350,  9.6, 0.75, 51, 1],
    [26, 420, 12.0, 1.05, 65, 1],
    [27, 268,  7.2, 0.40, 33, 1],
    [28, 385, 10.9, 0.88, 58, 1],
    [29, 332,  9.3, 0.67, 48, 1],
    [30, 408, 11.7, 0.98, 63, 1],
]

columns = [
    "Bridge_ID", "Load_kN", "Deflection_mm",
    "Crack_Width_mm", "Age_Years", "Structural_Risk"
]
df = pd.DataFrame(data, columns=columns)

print("\\nDATASET: 30 BRIDGE STRUCTURAL HEALTH MONITORING RECORDS")
print(df.to_string(index=False))
print("\\nClass distribution:")
print(df["Structural_Risk"].value_counts().rename(index={0: "Structurally sound", 1: "At-risk"}))

# Separate predictors (X) and target (y).
feature_cols = ["Load_kN", "Deflection_mm", "Crack_Width_mm", "Age_Years"]
X = df[feature_cols]
y = df["Structural_Risk"]

# Initial data visualization.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Load_kN", y="Deflection_mm",
    hue="Structural_Risk", style="Structural_Risk", s=100
)
plt.title("Bridge load vs. midspan deflection")
plt.xlabel("Live Load (kN)")
plt.ylabel("Deflection (mm)")
plt.legend(title="0 = Sound, 1 = At-risk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[feature_cols + ["Structural_Risk"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of structural health variables")
plt.tight_layout()
plt.show()

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

# Stratification preserves the class ratio in both subsets.
# With only 30 records, test metrics will be highly sensitive to individual records.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

# ============================================================
# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================

# SVM: scaling is important because SVM uses distances/margins.
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# Gaussian Naive Bayes: assumes feature values are conditionally Gaussian
# and conditionally independent given the class.
naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

# Decision Tree: learns a sequence of feature-based splitting rules.
decision_tree_model = DecisionTreeClassifier(
    max_depth=3, random_state=42
)

# Random Forest: combines many decision trees trained on bootstrap samples.
random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42
)

# AdaBoost: sequentially trains weak learners, emphasizing previously
# misclassified training examples.
# SAMME is explicitly selected for compatibility across sklearn versions.
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():
    print("\\n" + "=" * 65)
    print(f"{model_name.upper()}")
    print("=" * 65)

    # Learn from the training records.
    model.fit(X_train, y_train)

    # Predict labels for unseen test records.
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    # Calculate evaluation metrics.
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
    print("\\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Structurally sound", "At-risk"],
        zero_division=0
    ))

    # Confusion matrix visualization for this algorithm.
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Sound", "At-risk"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Comparison of five supervised learning algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 6. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values)
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Structural health feature")
    plt.tight_layout()
    plt.show()

# Visualize the learned Decision Tree.
plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Structurally sound", "At-risk"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree learned from bridge structural health training records")
plt.tight_layout()
plt.show()

# ============================================================
# PART 7. TEST A NEW ENGINEERING OBSERVATION
# ============================================================

# Example values are illustrative. They are not a real structural assessment —
# only for you (Software Engineering 4th year students).
new_bridge = pd.DataFrame([{
    "Load_kN": 355,
    "Deflection_mm": 9.9,
    "Crack_Width_mm": 0.78,
    "Age_Years": 52
}])

print("\\nNEW BRIDGE OBSERVATION")
print(new_bridge)

for name, model in models.items():
    predicted_class = int(model.predict(new_bridge)[0])
    label = "At-risk" if predicted_class == 1 else "Structurally sound"
    print(f"{name:15s} prediction: {label}")

print("\\nEND OF PRACTICAL CASE STUDY")

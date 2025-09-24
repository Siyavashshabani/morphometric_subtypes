import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import difflib

# -----------------------
# 1) Load data (robust sheet match)
# -----------------------
path = "data/expression_with_clusters.xlsx"
target = "prepared"  # intended name; file uses "prepapred"

xls = pd.ExcelFile(path)
names = xls.sheet_names
norm = {s.strip().lower(): s for s in names}

if target in norm:
    sheet = norm[target]
else:
    cand = difflib.get_close_matches(target, list(norm.keys()), n=1, cutoff=0.5)
    if not cand:
        raise ValueError(f'No sheet close to "{target}". Available: {names}')
    sheet = norm[cand[0]]

print("Loading sheet:", sheet)
df = pd.read_excel(path, sheet_name=sheet)

# -----------------------
# 2) X (genes) and y (labels)
# -----------------------
X = df.iloc[:, 2:].copy()          # all gene columns
y = df["Cluster"].values           # labels (1..4)

print("X shape:", X.shape, "y shape:", y.shape)

# =========================================================
# A) Validation accuracy: split -> train on train -> eval on val
# =========================================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

dt_val = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)
dt_val.fit(X_tr, y_tr)

y_val_pred = dt_val.predict(X_val)
acc  = accuracy_score(y_val, y_val_pred)
bacc = balanced_accuracy_score(y_val, y_val_pred)

print("\n[VALIDATION] accuracy:", f"{acc:.3f}")
print("[VALIDATION] balanced accuracy:", f"{bacc:.3f}")
print("\n[VALIDATION] confusion matrix:\n", confusion_matrix(y_val, y_val_pred))
print("\n[VALIDATION] classification report:\n", classification_report(y_val, y_val_pred, digits=3))

# Optional: plot the trained validation tree (top depth only)
os.makedirs("pics", exist_ok=True)
plt.figure(figsize=(22, 12))
plot_tree(
    dt_val,
    feature_names=X.columns,
    class_names=[str(c) for c in np.unique(y_tr)],
    filled=True, impurity=False, proportion=True,
    max_depth=3, fontsize=10
)
plt.tight_layout()
plt.savefig("pics/decision_tree_val_depth3.png", dpi=300, bbox_inches="tight")
print("Saved tree plot: pics/decision_tree_val_depth3.png")

# =========================================================
# B) Feature importance on full data (your original part)
#    (If you prefer no leakage at all, compute importances on TRAIN instead.)
# =========================================================
dt_full = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)
dt_full.fit(X, y)

imp_series = pd.Series(dt_full.feature_importances_, index=X.columns).sort_values(ascending=False)
TOP_K = 30
print("\nTop genes by in-sample DT importance (full data):")
print(imp_series.head(TOP_K))

imp_series.to_csv("dt_importance_full_data.csv")
print('Saved: dt_importance_full_data.csv')

# Optional: text rules preview for the validation-trained tree
rules = export_text(dt_val, feature_names=list(X.columns), max_depth=3, decimals=3)
print("\nTop-of-tree rules (validation-trained model):\n", rules)

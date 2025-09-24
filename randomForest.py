import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# -----------------------
# 1) Load data
# -----------------------

path = "data/expression_with_clusters.xlsx"

# Inspect available sheets
xls = pd.ExcelFile(path)
print("Sheets found:", xls.sheet_names)

# Find a sheet exactly named "prepared" (case-insensitive, trims spaces)
target = "prepared"
match = next((s for s in xls.sheet_names if s.strip().lower() == target), None)

if match is None:
    raise ValueError(f'No sheet named "{target}" (case-insensitive) found. Available: {xls.sheet_names}')

# Load the matched sheet into a DataFrame
df = pd.read_excel(path, sheet_name=match)   # engine auto-detected; requires openpyxl for .xlsx
print(df.head())

X = df.iloc[:, 2:].copy()
y = df["Cluster"].values

print("X shape:", X.shape, "y shape:", y.shape)

# -----------------------
# 2) Hold-out split (no leakage)
# -----------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# 3) CV-based feature importance on TRAIN ONLY
# -----------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
feat_importances = np.zeros(X_tr.shape[1], dtype=float)
val_scores = []

# A slightly regularized RF to avoid overfitting with small n, many p
rf_base = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    oob_score=False,
    n_jobs=-1,
    random_state=42,
)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_tr, y_tr), 1):
    X_tr_cv, X_va_cv = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
    y_tr_cv, y_va_cv = y_tr[tr_idx], y_tr[va_idx]

    rf = rf_base
    rf.fit(X_tr_cv, y_tr_cv)

    # accumulate impurity-based importances
    feat_importances += rf.feature_importances_

    # track validation performance (balanced acc is informative with class imbalance)
    y_va_pred = rf.predict(X_va_cv)
    val_scores.append(balanced_accuracy_score(y_va_cv, y_va_pred))

feat_importances /= cv.get_n_splits()
cv_bacc = np.mean(val_scores)
print(f"[Train-CV] mean balanced accuracy over folds: {cv_bacc:.3f}")

# -----------------------
# 4) Pick top-k genes from TRAIN-CV importances
# -----------------------
TOP_K = 30  # tweak to 10/20/50 etc.
imp_series = pd.Series(feat_importances, index=X_tr.columns).sort_values(ascending=False)
top_genes = imp_series.head(TOP_K).index.tolist()

print("\nTop genes by CV-averaged RF importance:")
print(imp_series.head(20))   # show top 20 for quick glance

# -----------------------
# 5) Final model on TRAIN, evaluate on TEST (using top-k only)
# -----------------------
rf_final = RandomForestClassifier(
    n_estimators=800,
    max_features="sqrt",
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=123,
)
rf_final.fit(X_tr[top_genes], y_tr)

y_te_pred = rf_final.predict(X_te[top_genes])
acc = accuracy_score(y_te, y_te_pred)
bacc = balanced_accuracy_score(y_te, y_te_pred)

print("\n[TEST] accuracy:", f"{acc:.3f}")
print("[TEST] balanced accuracy:", f"{bacc:.3f}")
print("\n[TEST] confusion matrix:\n", confusion_matrix(y_te, y_te_pred))
print("\n[TEST] classification report:\n", classification_report(y_te, y_te_pred, digits=3))

# -----------------------
# 6) Permutation importance on TEST (for top-k genes)
# (more reliable than impurity importance; computed on untouched test set)
# -----------------------
perm = permutation_importance(
    rf_final, X_te[top_genes], y_te,
    n_repeats=30, random_state=7, n_jobs=-1,
    scoring="balanced_accuracy"
)

perm_df = (
    pd.DataFrame({
        "gene": top_genes,
        "mean_importance": perm.importances_mean,
        "std_importance": perm.importances_std
    })
    .sort_values("mean_importance", ascending=False)
    .reset_index(drop=True)
)

print("\n[TEST] Permutation importance (scoring=balanced_accuracy) for top-k genes:")
print(perm_df.head(20))

# Optional: save ranked lists
imp_series.to_csv("rf_cv_importance_train.csv")    # all genes ranked (train-CV)
perm_df.to_csv("rf_perm_importance_test_topk.csv") # top-k genes ranked (test permutation)
print("\nSaved: rf_cv_importance_train.csv, rf_perm_importance_test_topk.csv")


import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

# Pick one tree from the trained forest (e.g., rf_final from earlier)
est = rf_final.estimators_[1]  # try 0, 1, 2, ...

plt.figure(figsize=(24, 12))
tree.plot_tree(
    est,
    feature_names=top_genes,                            # the features you trained rf_final with
    class_names=[str(c) for c in np.unique(y_tr)],     # label names
    filled=True,
    impurity=False,
    proportion=True,
    max_depth=3,      # <-- keep small for readability; try 3–4
    fontsize=10
)
plt.tight_layout()
plt.show()

# (Optional) Save to file
plt.savefig("pics/rf_tree0_depth3.png", dpi=300, bbox_inches="tight")

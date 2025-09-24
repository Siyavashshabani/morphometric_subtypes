import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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
print(df.head())

# -----------------------
# 2) X (genes) and y (labels)
# -----------------------
X = df.iloc[:, 2:].copy()          # all gene columns
y = df["Cluster"].values           # labels (1..4)

print("X shape:", X.shape, "y shape:", y.shape)

# -----------------------
# 3) Fit ONE tree on ALL data, get importances
# -----------------------
dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,            # shallow to reduce variance
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)
dt.fit(X, y)

imp_series = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
TOP_K = 30
print("\nTop genes by in-sample DT importance:")
print(imp_series.head(TOP_K))

# Save full ranking
imp_series.to_csv("dt_importance_full_data.csv")
print('\nSaved: dt_importance_full_data.csv')

# -----------------------
# 4) (Optional) Plot a shallow view of the tree
# -----------------------
plt.figure(figsize=(22, 12))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=[str(c) for c in np.unique(y)],
    filled=True,
    impurity=False,
    proportion=True,
    max_depth=3,   # visualize only the top of the tree
    fontsize=10
)
plt.tight_layout()
plt.savefig("pics/decision_tree_full_depth3.png", dpi=300, bbox_inches="tight")
print("Saved tree plot: decision_tree_full_depth3.png")

# (Optional) Human-readable rules for top of tree
rules = export_text(dt, feature_names=list(X.columns), max_depth=3, decimals=3)
print("\nTop-of-tree rules:\n", rules)

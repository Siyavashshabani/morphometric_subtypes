import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

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


# Define features (genes only) and labels
X = df.iloc[:, 2:]        # all gene columns
y = df["Cluster"].values  # labels

# 1) One train/validation split (no cross_val_score)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 2) Fit once per gene on TRAIN, evaluate once on VAL
rows = []
for gene in X.columns:
    Xi_tr  = X_tr[[gene]]
    Xi_val = X_val[[gene]]
    clf = LogisticRegression(
        solver="lbfgs", #multi_class="multinomial", 
        max_iter=1000, class_weight="balanced"
    )
    clf.fit(Xi_tr, y_tr)
    y_hat = clf.predict(X_val[[gene]])
    acc   = accuracy_score(y_val, y_hat)
    bacc  = balanced_accuracy_score(y_val, y_hat)
    rows.append({"gene": gene, "val_accuracy": acc, "val_bal_accuracy": bacc})


    
# 3) Rank by validation balanced accuracy (more robust w/ class imbalance)
ranking_df = pd.DataFrame(rows).sort_values(
    ["val_bal_accuracy", "val_accuracy"], ascending=False
).reset_index(drop=True)

print("Top 20 genes by validation balanced accuracy:")
print(ranking_df.head(20))

# Save results
ranking_df.to_csv("one_split_gene_logreg_accuracy.csv", index=False, float_format="%.6f")



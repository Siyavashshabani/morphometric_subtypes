import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

path = "data/expression_with_clusters.xlsx"

# Inspect available sheets
xls = pd.ExcelFile(path)
print("Sheets found:", xls.sheet_names)
target = "prepared"
match = next((s for s in xls.sheet_names if s.strip().lower() == target), None)
if match is None:
    raise ValueError(f'No sheet named "{target}" (case-insensitive) found. Available: {xls.sheet_names}')

# Load data
df = pd.read_excel(path, sheet_name=match)
print(df.head())

# Features & labels
X = df.iloc[:, 2:]        # all gene columns
y = df["Cluster"].values  # labels (4 classes)

# Single split reused across all one-vs-rest tasks
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ------- One-vs-Rest per class, per gene -------
all_rows = []
classes = np.unique(y)

for cls in classes:
    print(f"\n=== Target class: {cls} (one-vs-rest) ===")
    # Binarize labels for this class
    y_tr_bin  = (y_tr == cls).astype(int)
    y_val_bin = (y_val == cls).astype(int)

    rows = []
    for gene in X.columns:
        Xi_tr  = X_tr[[gene]]
        Xi_val = X_val[[gene]]

        clf = LogisticRegression(
            solver="lbfgs", max_iter=1000, class_weight="balanced"
        )
        clf.fit(Xi_tr, y_tr_bin)

        y_hat = clf.predict(Xi_val)
        acc   = accuracy_score(y_val_bin, y_hat)
        bacc  = balanced_accuracy_score(y_val_bin, y_hat)

        rows.append({
            "target_class": cls,
            "gene": gene,
            "val_accuracy": acc,
            "val_bal_accuracy": bacc,
            "val_pos_prevalence": float(y_val_bin.mean())  # just for context
        })

    ranking_df = pd.DataFrame(rows).sort_values(
        ["val_bal_accuracy", "val_accuracy"], ascending=False
    ).reset_index(drop=True)

    # Print top 10 for this class
    print(ranking_df.head(10))

    # Save per-class CSV
    safe_cls = str(cls).replace("/", "_")
    ranking_df.to_csv(f"ovr_gene_logreg_accuracy_class_{safe_cls}.csv",
                      index=False, float_format="%.6f")

    # Accumulate for combined file
    all_rows.extend(rows)

# Save combined results across all classes
combined_df = pd.DataFrame(all_rows).sort_values(
    ["target_class", "val_bal_accuracy", "val_accuracy"], ascending=[True, False, False]
).reset_index(drop=True)

combined_df.to_csv("ovr_gene_logreg_accuracy_all_classes.csv",
                   index=False, float_format="%.6f")

print("\nSaved:")
print(" - ovr_gene_logreg_accuracy_all_classes.csv")
print(" - ovr_gene_logreg_accuracy_class_<CLASS>.csv (one per class)")

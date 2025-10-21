# Let's load the user's results and compute 95% confidence intervals for accuracy per split
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint

# Load file
path = Path("full_evaluation_results_cleaned.csv")
df = pd.read_csv(path)

# Inspect columns and do some light cleaning
df.columns = [c.strip().lower() for c in df.columns]

# Try to infer split from the source_file column
split_col = None
for cand in ["source_file", "file", "split", "dataset", "subset"]:
    if cand in df.columns:
        split_col = cand
        break

# If we don't have a split column, create one as "ALL"
if split_col is None:
    df["split"] = "ALL"
    split_col = "split"

# Normalize predicted/ground truth columns
pred_col = None
gt_col = None
for cand in ["predicted_answer", "pred", "prediction", "model_answer", "y_pred"]:
    if cand in df.columns:
        pred_col = cand
        break
for cand in ["ground_truth", "answerkey", "label", "target", "y_true"]:
    if cand in df.columns:
        gt_col = cand
        break

if pred_col is None or gt_col is None:
    raise RuntimeError(f"Could not find prediction/ground-truth columns. Columns found: {df.columns.tolist()}")

# Extract split label (Train/Dev/Test) from source_file if present
def extract_split(val: str) -> str:
    val_lower = str(val).lower()
    if "train" in val_lower:
        return "Train"
    if "dev" in val_lower or "valid" in val_lower or "val" in val_lower:
        return "Dev"
    if "test" in val_lower:
        return "Test"
    return val

df["split_name"] = df[split_col].apply(extract_split)

# Compute correct flag
df["correct"] = (df[pred_col].astype(str).str.strip().str.upper() == df[gt_col].astype(str).str.strip().str.upper())

# Group and compute counts, accuracy, and 95% CI (Wilson and Clopper-Pearson)
rows = []
for split, g in df.groupby("split_name"):
    n = len(g)
    k = int(g["correct"].sum())
    acc = k / n if n else np.nan

    # Wilson 95% CI
    low_w, high_w = proportion_confint(k, n, alpha=0.05, method="wilson")
    # Clopper-Pearson exact 95% CI
    low_cp, high_cp = proportion_confint(k, n, alpha=0.05, method="beta")

    rows.append({
        "Split": split,
        "N": n,
        "Correct": k,
        "Accuracy": acc,
        "95% CI (Wilson) Low": low_w,
        "95% CI (Wilson) High": high_w,
        "95% CI (Exact) Low": low_cp,
        "95% CI (Exact) High": high_cp,
    })

summary = pd.DataFrame(rows).sort_values("Split")

# Overall as well
n_all = len(df)
k_all = int(df["correct"].sum())
acc_all = k_all / n_all if n_all else np.nan
low_w_all, high_w_all = proportion_confint(k_all, n_all, alpha=0.05, method="wilson")
low_cp_all, high_cp_all = proportion_confint(k_all, n_all, alpha=0.05, method="beta")

overall = pd.DataFrame([{
    "Split": "Overall",
    "N": n_all,
    "Correct": k_all,
    "Accuracy": acc_all,
    "95% CI (Wilson) Low": low_w_all,
    "95% CI (Wilson) High": high_w_all,
    "95% CI (Exact) Low": low_cp_all,
    "95% CI (Exact) High": high_cp_all,
}])

summary_full = pd.concat([summary, overall], ignore_index=True)

# Round for display
summary_display = summary_full.copy()
for col in ["Accuracy", "95% CI (Wilson) Low", "95% CI (Wilson) High", "95% CI (Exact) Low", "95% CI (Exact) High"]:
    summary_display[col] = (summary_display[col] * 100).round(2)

# Save to CSV for the manuscript supplement
out_path = Path("arc_accuracy_with_95CIs.csv")
summary_full.to_csv(out_path, index=False)

out_path, summary_display


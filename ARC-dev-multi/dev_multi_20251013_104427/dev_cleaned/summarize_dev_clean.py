# save as: summarize_dev_clean.py
import argparse
import pandas as pd
import math

# try exact CIs via statsmodels if available
try:
    from statsmodels.stats.proportion import proportion_confint
    HAS_SM = True
except Exception:
    HAS_SM = False

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + (z**2)/n
    center = p + (z**2)/(2*n)
    adj = z * math.sqrt((p*(1-p)/n) + (z**2)/(4*n**2))
    lo = (center - adj) / denom
    hi = (center + adj) / denom
    return (max(0.0, lo), min(1.0, hi))

def exact_ci(k, n, alpha=0.05):
    if not HAS_SM or n == 0:
        return (float("nan"), float("nan"))
    lo, hi = proportion_confint(k, n, alpha=alpha, method="beta")
    return (lo, hi)

def summarize(df, label):
    # prefer 'correct_clean', else fall back to 'correct'
    if "correct_clean" in df.columns:
        corr = df["correct_clean"].astype(int)
    else:
        corr = df["correct"].astype(int)
    n = len(df)
    k = int(corr.sum())
    acc = (k / n) if n else float("nan")
    wlo, whi = wilson_ci(k, n)
    elo, ehi = exact_ci(k, n)
    return {
        "run": label,
        "N": n,
        "Correct": k,
        "Accuracy": acc,
        "95% CI (Wilson) Low": wlo,
        "95% CI (Wilson) High": whi,
        "95% CI (Exact) Low": elo,
        "95% CI (Exact) High": ehi,
    }

def main():
    ap = argparse.ArgumentParser(description="Summarize cleaned Dev predictions with 95% CIs.")
    ap.add_argument("--input", default="dev_runs_all_rows_clean.csv", help="path to cleaned combined CSV")
    ap.add_argument("--output", default="dev_runs_summary_clean.csv", help="where to write the summary CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str)
    # coerce columns
    for col in ["run", "correct", "correct_clean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # group by run
    rows = []
    for r, sub in df.groupby("run", dropna=False):
        rows.append(summarize(sub, int(r) if pd.notna(r) else "NA"))

    # overall row
    rows.append(summarize(df, "ALL"))

    out = pd.DataFrame(rows)
    # pretty rounding
    for c in ["Accuracy", "95% CI (Wilson) Low", "95% CI (Wilson) High",
              "95% CI (Exact) Low", "95% CI (Exact) High"]:
        out[c] = out[c].astype(float).round(6)

    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()


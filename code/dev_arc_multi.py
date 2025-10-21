#!/usr/bin/env python3
"""
Dev-only multi-run evaluator for the SM agent on ARC-Challenge-Dev.

- Reads ONE CSV:  ~/Desktop/Codes/SM_Paper/ai2_benchmark/ARC-Challenge-Dev.csv
- Repeats the evaluation multiple times (N runs)
- For each run:
    * For each question: write user_prompt.txt -> call socrates.py -> read output.txt
    * Extract final answer letter via your Argo endpoint (A/B/C/D)
    * Record predictions and correctness
    * Compute per-run accuracy and 95% CIs (Wilson + Exact / Clopper–Pearson)
- After all runs:
    * Save per-run summaries, an across-run aggregate (mean ± SD and a 95% CI across runs),
      and all per-question rows across all runs.
- Never overwrites old results: creates /Users/hharb/Desktop/Codes/SM_Paper/ARC-dev-multi/<timestamp>/

Requirements in CWD:
  - api.txt (Argo URL), prompt.txt (system prompt used by socrates.py),
    letter_extraction_prompt.txt (for letter extraction), socrates.py
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import subprocess
import requests
import os
from datetime import datetime
from statsmodels.stats.proportion import proportion_confint
from math import sqrt

# ------------------------
# Utilities
# ------------------------

def load_text(path: Path, required: bool = True, encoding="utf-8") -> str:
    try:
        return path.read_text(encoding=encoding).strip()
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return ""

def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def run_socrates(socrates_py: Path):
    # Calls your existing socrates.py (which sends prompt.txt + user_prompt.txt to Argo and writes output.txt)
    subprocess.run(["python3", str(socrates_py)], check=True)

def extract_letter(api_url: str, letter_prompt: str, response_text: str,
                   temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 10) -> str:
    """Use your letter_extraction_prompt to extract a single letter A/B/C/D."""
    payload = {
        "user": "hharb",
        "model": "gpt4o",
        "system": letter_prompt,
        "prompt": [response_text],
        "stop": [],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=120)
        r.raise_for_status()
        letter = r.json().get("response", "").strip().upper()
        return letter if letter in ["A", "B", "C", "D"] else "?"
    except Exception as e:
        print(f"❌ Letter extraction failed: {e}")
        return "?"

def binomial_cis(k: int, n: int, alpha: float=0.05):
    """Return (Wilson_low, Wilson_high), (Exact_low, Exact_high)."""
    low_w, high_w = proportion_confint(k, n, alpha=alpha, method="wilson")
    low_cp, high_cp = proportion_confint(k, n, alpha=alpha, method="beta")
    return (low_w, high_w), (low_cp, high_cp)

def mean_sd_ci(values, alpha: float=0.05):
    """Mean, SD, and 95% CI across runs (normal approx; n>=5 recommended)."""
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    z = 1.96
    half = z * sd / (len(arr) ** 0.5) if len(arr) > 1 else 0.0
    return mean, sd, (mean - half, mean + half)

def normalize_columns(df: pd.DataFrame):
    # Case-insensitive access for 'question' and 'AnswerKey'
    lower = {c.lower(): c for c in df.columns}
    q_col = lower.get("question")
    a_col = lower.get("answerkey")
    if q_col is None or a_col is None:
        raise RuntimeError(f"CSV must include 'question' and 'AnswerKey' columns. Found: {df.columns.tolist()}")
    return q_col, a_col

# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser(description="Dev-only multi-run evaluator on ARC-Challenge-Dev.")
    parser.add_argument("--csv_path", type=str,
                        default=str(Path.home() / "Desktop/Codes/SM_Paper/ai2_benchmark/ARC-Challenge-Dev.csv"),
                        help="Path to ARC-Challenge-Dev.csv")
    parser.add_argument("--out_root", type=str,
                        default="/Users/hharb/Desktop/Codes/SM_Paper/ARC-dev-multi",
                        help="Root output folder (a timestamped subfolder will be created here)")
    parser.add_argument("--runs", type=int, default=5, help="Number of replications")
    parser.add_argument("--socrates_py", type=str, default="socrates.py", help="Path to your socrates.py")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds to allow socrates.py per question")
    args = parser.parse_args()

    # Resolve paths
    csv_path = Path(args.csv_path).expanduser().resolve()
    out_root = Path(args.out_root).resolve()
    socrates_py = Path(args.socrates_py).resolve()

    # Sanity checks
    if not csv_path.exists():
        raise FileNotFoundError(f"ARC Dev CSV not found: {csv_path}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Load required side files from CWD
    cwd = Path(".").resolve()
    api_url = load_text(cwd / "api.txt")
    letter_prompt = load_text(cwd / "letter_extraction_prompt.txt", required=False)
    if not letter_prompt:
        # Minimal fallback if the file is missing
        letter_prompt = "You will be given a model's full reasoning and a multiple-choice question.\nExtract ONLY the final chosen option letter (A, B, C, or D). Respond with a single uppercase letter."
        print("⚠️  'letter_extraction_prompt.txt' not found. Using a minimal built-in prompt.")

    # Timestamped output directory; never overwrite
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"dev_multi_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the Dev CSV
    # Use pandas robust parsing (quoted fields handled by default)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    q_col, a_col = normalize_columns(df)

    # Prepare outputs
    all_rows = []
    per_run = []

    # Log raw outputs per run (optional, can be large)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Per-run loop
    for run_idx in range(1, args.runs + 1):
        print(f"\n=== Run {run_idx}/{args.runs} ===")
        total = 0
        correct = 0
        per_q_rows = []
        log_path = logs_dir / f"dev_run_{run_idx:02d}.txt"

        with log_path.open("w", encoding="utf-8") as logf:
            for i, row in df.iterrows():
                question = str(row[q_col])
                gt = str(row[a_col]).strip().upper()

                # Write user prompt
                write_text(cwd / "user_prompt.txt", question)

                # Call your existing socrates.py (it will read prompt files and write output.txt)
                try:
                    subprocess.run(["python3", str(socrates_py)], check=True, timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    print(f"⏱️  Timeout at question #{i+1}; recording '?' and continuing.")
                    pred = "?"
                    is_correct = (pred == gt)
                    per_q_rows.append({
                        "run": run_idx,
                        "question_number": i + 1,
                        "predicted_answer": pred,
                        "ground_truth": gt,
                        "correct": int(is_correct),
                    })
                    continue

                # Read the full response and extract the final letter
                response_text = load_text(cwd / "output.txt", required=False)
                pred = extract_letter(api_url, letter_prompt, response_text, temperature=0.0, top_p=1.0, max_tokens=10)

                # Log the raw model output for traceability
                logf.write(f"--- Q{ i+1 } ---\n")
                logf.write(response_text + "\n\n")

                total += 1
                is_correct = (pred == gt)
                if is_correct:
                    correct += 1

                per_q_rows.append({
                    "run": run_idx,
                    "question_number": i + 1,
                    "predicted_answer": pred,
                    "ground_truth": gt,
                    "correct": int(is_correct),
                })

        # Save per-run predictions
        per_run_pred_path = out_dir / f"dev_run_{run_idx:02d}_predictions.csv"
        pd.DataFrame(per_q_rows).to_csv(per_run_pred_path, index=False)

        # Per-run summary with CIs (over items)
        n = total
        k = correct
        acc = (k / n) if n else float("nan")
        (low_w, high_w), (low_cp, high_cp) = binomial_cis(k, n, alpha=0.05)

        per_run.append({
            "run": run_idx,
            "N": n,
            "Correct": k,
            "Accuracy": acc,
            "95% CI (Wilson) Low": low_w,
            "95% CI (Wilson) High": high_w,
            "95% CI (Exact) Low": low_cp,
            "95% CI (Exact) High": high_cp,
        })

        all_rows.extend(per_q_rows)

    # Save per-run summary table
    per_run_df = pd.DataFrame(per_run)
    per_run_df.to_csv(out_dir / "dev_runs_summary.csv", index=False)

    # Across-run mean ± SD and a 95% CI across runs (normal approx)
    mean_acc, sd_acc, (ci_low, ci_high) = mean_sd_ci(per_run_df["Accuracy"].tolist())
    agg_df = pd.DataFrame([{
        "Runs": args.runs,
        "Mean Accuracy": mean_acc,
        "SD Accuracy": sd_acc,
        "95% CI over runs (normal approx) Low": ci_low,
        "95% CI over runs (normal approx) High": ci_high,
    }])
    agg_df.to_csv(out_dir / "dev_runs_aggregate.csv", index=False)

    # Save all rows together
    pd.DataFrame(all_rows).to_csv(out_dir / "dev_runs_all_rows.csv", index=False)

    # README with configuration
    readme = f"""Dev multi-run evaluation on ARC-Challenge-Dev

CSV: {csv_path}
Runs: {args.runs}
Working dir: {cwd}
socrates.py: {socrates_py}

Artifacts:
- Per-run predictions: dev_run_XX_predictions.csv
- Per-run summaries:   dev_runs_summary.csv  (includes Wilson & Exact 95% CIs over items)
- Across-run aggregate: dev_runs_aggregate.csv  (mean ± SD and a 95% CI across runs)
- Combined rows:       dev_runs_all_rows.csv
- Raw outputs:         logs/dev_run_XX.txt

Notes:
- CIs over items reflect single-run uncertainty across questions, not between-run randomness.
- If your API supports true seeds, add them to socrates.py payload.
"""
    write_text(out_dir / "README.txt", readme)

    print("\n✅ Done.")
    print(f"📁 Results directory: {out_dir}")
    print(f"📄 Per-run summary:   {out_dir / 'dev_runs_summary.csv'}")
    print(f"📄 Across-run aggregate: {out_dir / 'dev_runs_aggregate.csv'}")
    print(f"🗒️  Logs in:          {out_dir / 'logs'}")

if __name__ == "__main__":
    main()


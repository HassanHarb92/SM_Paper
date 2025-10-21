#!/usr/bin/env python3
"""
Test-only multi-run evaluator for the SM agent on ARC-Challenge-Test.

- Reads ONE CSV:  ~/Desktop/Codes/SM_Paper/ai2_benchmark/ARC-Challenge-Test.csv
- Same behavior as the Train/Dev runners.
- Never overwrites old results: creates /Users/hharb/Desktop/Codes/SM_Paper/ARC-test-multi/<timestamp>/
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import subprocess
import requests
from datetime import datetime
from statsmodels.stats.proportion import proportion_confint

def load_text(path: Path, required: bool = True, encoding="utf-8") -> str:
    try:
        return path.read_text(encoding=encoding).strip()
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return ""

def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def extract_letter(api_url: str, letter_prompt: str, response_text: str,
                   temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 10) -> str:
    payload = {
        "user": "answer_extractor",
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
        print(f"Letter extraction failed: {e}")
        return "?"

def binomial_cis(k: int, n: int, alpha: float=0.05):
    low_w, high_w = proportion_confint(k, n, alpha=alpha, method="wilson")
    low_cp, high_cp = proportion_confint(k, n, alpha=alpha, method="beta")
    return (low_w, high_w), (low_cp, high_cp)

def mean_sd_ci(values, alpha: float=0.05):
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    z = 1.96
    half = z * sd / (len(arr) ** 0.5) if len(arr) > 1 else 0.0
    return mean, sd, (mean - half, mean + half)

def normalize_columns(df: pd.DataFrame):
    lower = {c.lower(): c for c in df.columns}
    q_col = lower.get("question")
    a_col = lower.get("answerkey")
    if q_col is None or a_col is None:
        raise RuntimeError(f"CSV must include 'question' and 'AnswerKey' columns. Found: {df.columns.tolist()}")
    return q_col, a_col

def main():
    parser = argparse.ArgumentParser(description="Test-only multi-run evaluator on ARC-Challenge-Test.")
    parser.add_argument("--csv_path", type=str,
                        default=str(Path.home() / "Desktop/Codes/SM_Paper/ai2_benchmark/ARC-Challenge-Test.csv"))
    parser.add_argument("--out_root", type=str,
                        default="/Users/hharb/Desktop/Codes/SM_Paper/ARC-test-multi")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--socrates_py", type=str, default="socrates.py")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    out_root = Path(args.out_root).resolve()
    socrates_py = Path(args.socrates_py).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"ARC Test CSV not found: {csv_path}")
    out_root.mkdir(parents=True, exist_ok=True)

    cwd = Path(".").resolve()
    api_url = load_text(cwd / "api.txt")
    letter_prompt = load_text(cwd / "letter_extraction_prompt.txt", required=False) or \
        "You will be given a model's full reasoning and a multiple-choice question.\nExtract ONLY the final chosen option letter (A, B, C, or D). Respond with a single uppercase letter."

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"test_multi_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    q_col, a_col = normalize_columns(df)

    all_rows, per_run = [], []

    for run_idx in range(1, args.runs + 1):
        total = 0
        correct = 0
        per_q_rows = []
        log_path = logs_dir / f"test_run_{run_idx:02d}.txt"

        with log_path.open("w", encoding="utf-8") as logf:
            for i, row in df.iterrows():
                question = str(row[q_col])
                gt = str(row[a_col]).strip().upper()

                write_text(cwd / "user_prompt.txt", question)

                try:
                    subprocess.run(["python3", str(socrates_py)], check=True, timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    pred = "?"
                    is_correct = (pred == gt)
                    per_q_rows.append({"run": run_idx, "question_number": i + 1,
                                       "predicted_answer": pred, "ground_truth": gt, "correct": int(is_correct)})
                    continue

                response_text = load_text(cwd / "output.txt", required=False)
                pred = extract_letter(api_url, letter_prompt, response_text, 0.0, 1.0, 10)

                logf.write(f"--- Q{ i+1 } ---\n{response_text}\n\n")

                total += 1
                is_correct = (pred == gt)
                if is_correct:
                    correct += 1

                per_q_rows.append({"run": run_idx, "question_number": i + 1,
                                   "predicted_answer": pred, "ground_truth": gt, "correct": int(is_correct)})

        pd.DataFrame(per_q_rows).to_csv(out_dir / f"test_run_{run_idx:02d}_predictions.csv", index=False)

        n, k = total, correct
        acc = (k / n) if n else float("nan")
        (low_w, high_w), (low_cp, high_cp) = binomial_cis(k, n)
        per_run.append({"run": run_idx, "N": n, "Correct": k, "Accuracy": acc,
                        "95% CI (Wilson) Low": low_w, "95% CI (Wilson) High": high_w,
                        "95% CI (Exact) Low": low_cp, "95% CI (Exact) High": high_cp})
        all_rows.extend(per_q_rows)

    per_run_df = pd.DataFrame(per_run)
    per_run_df.to_csv(out_dir / "test_runs_summary.csv", index=False)

    mean_acc, sd_acc, (ci_low, ci_high) = mean_sd_ci(per_run_df["Accuracy"].tolist())
    pd.DataFrame([{"Runs": args.runs, "Mean Accuracy": mean_acc, "SD Accuracy": sd_acc,
                   "95% CI over runs (normal approx) Low": ci_low,
                   "95% CI over runs (normal approx) High": ci_high}]
                 ).to_csv(out_dir / "test_runs_aggregate.csv", index=False)

    pd.DataFrame(all_rows).to_csv(out_dir / "test_runs_all_rows.csv", index=False)

    readme = f"""Test multi-run evaluation on ARC-Challenge-Test

CSV: {csv_path}
Runs: {args.runs}
Working dir: {cwd}
socrates.py: {socrates_py}

Artifacts:
- test_run_XX_predictions.csv
- test_runs_summary.csv  (Wilson & Exact 95% CIs over items)
- test_runs_aggregate.csv  (mean ± SD and a 95% CI across runs)
- test_runs_all_rows.csv
- logs/test_run_XX.txt
"""
    write_text(out_dir / "README.txt", readme)
    print("Done.")
    print(f"Results directory: {out_dir}")

if __name__ == "__main__":
    main()


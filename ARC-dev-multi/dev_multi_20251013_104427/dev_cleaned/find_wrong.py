# save as: find_all_wrong_from_clean_runs.py
import argparse, glob, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Find questions wrong in ALL Dev runs (using *_clean.csv files).")
    ap.add_argument("--dir", default=".", help="directory with dev_run_0X_predictions_clean.csv files")
    ap.add_argument("--pattern", default="dev_run_0*_predictions_clean.csv",
                    help="glob pattern for per-run cleaned files")
    ap.add_argument("--output", default="dev_questions_all_wrong_across_runs.csv",
                    help="output CSV with question_number list")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print("No files found. Check --dir and --pattern.")
        return

    wrong_sets = []
    file_labels = []

    for p in paths:
        df = pd.read_csv(p, dtype=str)
        if "question_number" not in df.columns:
            print(f"Skipping {os.path.basename(p)} (missing 'question_number').")
            continue

        # prefer 'correct_clean' if present
        corr_col = "correct_clean" if "correct_clean" in df.columns else "correct"
        if corr_col not in df.columns:
            print(f"Skipping {os.path.basename(p)} (missing '{corr_col}').")
            continue

        # coerce types
        df["question_number"] = pd.to_numeric(df["question_number"], errors="coerce").astype("Int64")
        df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce").fillna(0).astype(int)

        wrong_qs = set(df.loc[df[corr_col] == 0, "question_number"].dropna().astype(int).tolist())
        wrong_sets.append(wrong_qs)
        file_labels.append(os.path.basename(p))

        print(f"{os.path.basename(p)}: wrong={len(wrong_qs)}")

    if not wrong_sets:
        print("No usable files after checks.")
        return

    # intersection across all runs
    all_wrong = set.intersection(*wrong_sets) if len(wrong_sets) > 1 else wrong_sets[0]
    all_wrong_sorted = sorted(all_wrong)

    print("\nQuestions wrong in ALL runs:")
    if all_wrong_sorted:
        print(", ".join(map(str, all_wrong_sorted)))
    else:
        print("(none)")

    # write CSV
    out_df = pd.DataFrame({"question_number": all_wrong_sorted})
    out_path = os.path.join(args.dir, args.output)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} (count={len(out_df)})")

if __name__ == "__main__":
    main()


# save as: find_all_wrong_from_clean_runs.py
import argparse, glob, os, re
import pandas as pd

def get_run_id(df, path):
    # Prefer a single unique 'run' value if present
    if "run" in df.columns:
        vals = pd.to_numeric(df["run"], errors="coerce").dropna().unique()
        if len(vals) == 1:
            return int(vals[0])
    # Else parse from filename like dev_run_03_predictions_clean.csv
    m = re.search(r"dev_run_(\d+)_predictions", os.path.basename(path))
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot determine run id for file: {path}")

def main():
    ap = argparse.ArgumentParser(
        description="Find questions wrong in ALL Dev runs (using *_clean.csv files), and record GT + per-run answers."
    )
    ap.add_argument("--dir", default=".", help="directory with dev_run_0X_predictions_clean.csv files")
    ap.add_argument("--pattern", default="dev_run_0*_predictions_clean.csv",
                    help="glob pattern for per-run cleaned files")
    ap.add_argument("--output_list", default="dev_questions_all_wrong_across_runs.csv",
                    help="output CSV with question_number list")
    ap.add_argument("--output_matrix", default="dev_questions_all_wrong_with_preds.csv",
                    help="output CSV with GT and per-run predictions for the all-wrong items")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print("No files found. Check --dir and --pattern.")
        return

    wrong_sets = []
    run_preds = {}   # run_id -> Series(question_number -> prediction, as-is from file)
    gt_series = []   # list of Series(question_number -> GT, as-is from file)

    for p in paths:
        df = pd.read_csv(p, dtype=str)
        if "question_number" not in df.columns:
            print(f"Skipping {os.path.basename(p)} (missing 'question_number').")
            continue

        # pick correctness column (no recompute)
        corr_col = "correct_clean" if "correct_clean" in df.columns else "correct"
        if corr_col not in df.columns:
            print(f"Skipping {os.path.basename(p)} (missing '{corr_col}').")
            continue

        # pick prediction/GT columns (prefer normalized ones if present; do NOT remap here)
        pred_col = "pred_norm" if "pred_norm" in df.columns else "predicted_answer"
        gt_col   = "gt_norm"   if "gt_norm"   in df.columns else "ground_truth"

        run_id = get_run_id(df, p)

        # coerce types and keep one row per question_number
        df["question_number"] = pd.to_numeric(df["question_number"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["question_number"]).copy()
        df = df.sort_values(["question_number"]).drop_duplicates(subset=["question_number"], keep="first")

        # wrong set for this run
        df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce").fillna(0).astype(int)
        wrong_qs = set(df.loc[df[corr_col] == 0, "question_number"].dropna().astype(int).tolist())
        wrong_sets.append(wrong_qs)

        # store preds and GT as-is (already cleaned upstream)
        s_pred = pd.Series(df[pred_col].astype(str).values,
                           index=df["question_number"].astype(int), name=f"pred_run_{run_id:02d}")
        s_gt   = pd.Series(df[gt_col].astype(str).values,
                           index=df["question_number"].astype(int), name="ground_truth")
        run_preds[run_id] = s_pred
        gt_series.append(s_gt)

        print(f"{os.path.basename(p)}: wrong={len(wrong_qs)} | run={run_id}")

    if not wrong_sets:
        print("No usable files after checks.")
        return

    # intersection across all runs
    all_wrong = set.intersection(*wrong_sets) if len(wrong_sets) > 1 else wrong_sets[0]
    all_wrong_sorted = sorted(all_wrong)

    print("\nQuestions wrong in ALL runs:")
    print(", ".join(map(str, all_wrong_sorted)) if all_wrong_sorted else "(none)")

    # write simple list
    out_list_path = os.path.join(args.dir, args.output_list)
    pd.DataFrame({"question_number": all_wrong_sorted}).to_csv(out_list_path, index=False)
    print(f"\nWrote {out_list_path} (count={len(all_wrong_sorted)})")

    # build matrix for all-wrong items with GT + per-run predictions (no remapping)
    if all_wrong_sorted:
        # ground truth by simple mode across files (as-is)
        gt_df = pd.concat(gt_series, axis=1).T
        def mode_or_first(col):
            vc = col.value_counts(dropna=True)
            return vc.index[0] if not vc.empty else ""
        gt_mode = gt_df.apply(mode_or_first, axis=0)

        mat = pd.DataFrame({"question_number": all_wrong_sorted}).set_index("question_number")
        mat["ground_truth"] = gt_mode.reindex(mat.index)

        for run_id, s_pred in sorted(run_preds.items()):
            mat[f"pred_run_{run_id:02d}"] = s_pred.reindex(mat.index)

        out_matrix_path = os.path.join(args.dir, args.output_matrix)
        mat.reset_index().to_csv(out_matrix_path, index=False)
        print(f"Wrote {out_matrix_path} (rows={len(mat)})")

if __name__ == "__main__":
    main()


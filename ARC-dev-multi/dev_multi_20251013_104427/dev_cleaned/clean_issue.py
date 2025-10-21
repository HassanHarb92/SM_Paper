# save as: clean_dev_predictions.py
import argparse, glob, os
import pandas as pd

MAP = {"A":"A","1":"A","B":"B","2":"B","C":"C","3":"C","D":"D","4":"D","E":"E","5":"E"}
VALID = set("ABCDE")

def norm_choice(x: str) -> str:
    if x is None: return ""
    s = str(x).strip().upper()
    if s == "?": return "?"
    return MAP.get(s, s)

def main():
    ap = argparse.ArgumentParser(description="Clean Dev predictions: map letters↔numbers, interactively resolve '?', recompute correctness.")
    ap.add_argument("--dir", default=".", help="directory with dev_run_0X_predictions.csv files")
    ap.add_argument("--pattern", default="dev_run_0*_predictions.csv", help="glob pattern for Dev files")
    ap.add_argument("--logs_dir", default="logs", help="directory with logs/dev_run_XX.txt")
    ap.add_argument("--interactive", action="store_true", help="prompt to resolve '?' rows")
    ap.add_argument("--out_suffix", default="_clean", help="suffix for cleaned per-file outputs")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        print("No Dev prediction files found.")
        return

    overall_rows = []
    changed_rows = []

    print("\n=== Cleaning Dev predictions ===")
    for f in files:
        df = pd.read_csv(f, dtype=str)
        need = {"run","question_number","predicted_answer","ground_truth","correct"}
        if not need.issubset(df.columns):
            print(f"Skipping {os.path.basename(f)} (missing columns). Found: {df.columns.tolist()}")
            continue

        # Preserve original for audit
        df["original_predicted"] = df["predicted_answer"]

        # Normalize
        df["pred_norm"] = df["predicted_answer"].apply(norm_choice)
        df["gt_norm"]   = df["ground_truth"].apply(norm_choice)

        # Resolve unknowns interactively
        unknown_idx = df.index[df["pred_norm"]=="?"].tolist()
        if unknown_idx and args.interactive:
            print(f"\nResolving unknowns in {os.path.basename(f)}")
        for idx in unknown_idx:
            run = df.at[idx, "run"]
            qn  = df.at[idx, "question_number"]
            gt  = df.at[idx, "gt_norm"]
            try:
                log_name = f"dev_run_{int(run):02d}.txt"
            except:
                log_name = "dev_run_XX.txt"
            log_path = os.path.join(args.logs_dir, log_name)
            print(f"\nFile: {os.path.basename(f)} | Run: {run} | Q#: {qn} | GT: {gt or '(n/a)'}")
            print(f"Open log and check: {log_path}")
            user = input("Enter final answer (A–E or 1–5). Press Enter to skip: ").strip().upper()
            if user:
                fixed = norm_choice(user)
                if fixed in VALID:
                    df.at[idx, "pred_norm"] = fixed
                    print(f"Recorded: {fixed}")
                else:
                    print("Invalid entry. Kept '?'.")

        # Recompute correctness on normalized letters
        df["correct_clean"] = ((df["pred_norm"].isin(VALID)) & (df["pred_norm"] == df["gt_norm"])).astype(int)

        # Store cleaned predicted_answer as canonical letter when valid
        df["predicted_answer"] = df.apply(
            lambda r: r["pred_norm"] if r["pred_norm"] in VALID else r["predicted_answer"], axis=1
        )

        # Summaries
        n = len(df)
        acc_orig = (df["correct"].astype(int).sum()/n) if n else 0.0
        acc_clean = (df["correct_clean"].sum()/n) if n else 0.0
        unresolved = int((df["pred_norm"]=="?").sum())
        remapped = df[(df["original_predicted"]!=df["predicted_answer"]) | (df["correct"].astype(str)!=df["correct_clean"].astype(str))]

        # Save per-file cleaned CSV
        base, ext = os.path.splitext(f)
        out_path = base + args.out_suffix + ext
        df.to_csv(out_path, index=False)

        # Track overall and changed rows with file tag
        df_copy = df.copy()
        df_copy.insert(0, "file", os.path.basename(f))
        overall_rows.append(df_copy)

        if not remapped.empty:
            remap_copy = remapped.copy()
            remap_copy.insert(0, "file", os.path.basename(f))
            changed_rows.append(remap_copy[["file","run","question_number","original_predicted","predicted_answer","ground_truth","correct","correct_clean"]])

        print(f"\n{os.path.basename(f)}")
        print(f"  N: {n}")
        print(f"  Original accuracy: {acc_orig:.4f}")
        print(f"  Cleaned  accuracy: {acc_clean:.4f}")
        print(f"  Unresolved '?': {unresolved}")
        print(f"  Changed rows written to audit later.")

    # Write combined outputs
    if overall_rows:
        all_df = pd.concat(overall_rows, ignore_index=True)
        all_out = os.path.join(args.dir, "dev_runs_all_rows_clean.csv")
        all_df.to_csv(all_out, index=False)
        print(f"\nWrote combined rows: {all_out}")

    if changed_rows:
        ch_df = pd.concat(changed_rows, ignore_index=True)
        ch_out = os.path.join(args.dir, "changed_rows.csv")
        ch_df.to_csv(ch_out, index=False)
        print(f"Wrote audit of changed rows: {ch_out}")

    print("\nDone. Dev cleaning completed.")

if __name__ == "__main__":
    main()


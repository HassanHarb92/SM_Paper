# save as: print_dev_run_failures.py
import argparse, glob, os, csv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=".", help="directory with dev_run_0X_predictions.csv files")
    p.add_argument("--pattern", default="dev_run_0*_predictions.csv", help="glob pattern")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        print("No files found.")
        return

    for f in files:
        print(f"\n==== {os.path.basename(f)} ====")
        with open(f, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            failed = []
            for row in reader:
                try:
                    correct_flag = int(row.get("correct", "0"))
                except ValueError:
                    correct_flag = 0
                if correct_flag == 0:
                    failed.append(row)

        if not failed:
            print("No failures.")
            continue

        # print header once
        cols = ["run", "question_number", "predicted_answer", "ground_truth", "correct"]
        print(",".join(cols))
        for r in failed:
            print(",".join(str(r.get(c, "")).strip() for c in cols))

if __name__ == "__main__":
    main()


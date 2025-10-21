Dev multi-run evaluation on ARC-Challenge-Dev

CSV: /Users/hharb/Desktop/Codes/SM_Paper/ai2_benchmark/ARC-Challenge-Dev.csv
Runs: 5
Working dir: /Users/hharb/Desktop/Codes/SM_Paper/code
socrates.py: /Users/hharb/Desktop/Codes/SM_Paper/code/socrates.py

Artifacts:
- Per-run predictions: dev_run_XX_predictions.csv
- Per-run summaries:   dev_runs_summary.csv  (includes Wilson & Exact 95% CIs over items)
- Across-run aggregate: dev_runs_aggregate.csv  (mean ± SD and a 95% CI across runs)
- Combined rows:       dev_runs_all_rows.csv
- Raw outputs:         logs/dev_run_XX.txt

Notes:
- CIs over items reflect single-run uncertainty across questions, not between-run randomness.
- If your API supports true seeds, add them to socrates.py payload.

import pandas as pd
import subprocess
import requests
import os
from glob import glob

# === CONFIGURATION ===
DATA_DIR = "/Users/hharb/Desktop/Codes/AthensLLM/ai2_reasoning/ai2_data"
RESULT_CSV = "full_evaluation_results.csv"
SUMMARY_TXT = "summary_accuracy_report.txt"
LOG_TXT = "log_all.txt"

# Load API URL
with open("api.txt", "r") as f:
    API_URL = f.read().strip()

# Load letter extraction prompt
with open("letter_extraction_prompt.txt", "r", encoding="utf-8") as f:
    LETTER_PROMPT = f.read().strip()

def write_prompt_to_file(prompt):
    with open("user_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt.strip())

def run_socrates():
    subprocess.run(["python3", "socrates.py"], check=True)

def read_model_output():
    with open("output.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_letter(response_text):
    payload = {
        "user": "answer_extractor",
        "model": "gpt4o",
        "system": LETTER_PROMPT,
        "prompt": [response_text],
        "stop": [],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 10,
    }
    try:
        r = requests.post(API_URL, headers={"Content-Type": "application/json"}, json=payload)
        r.raise_for_status()
        letter = r.json().get("response", "").strip().upper()
        return letter if letter in ['A', 'B', 'C', 'D'] else "?"
    except Exception as e:
        print(f"❌ Letter extraction failed: {e}")
        return "?"

# === MAIN EVALUATION ===
all_results = []
summary_lines = []

with open(LOG_TXT, "w", encoding="utf-8") as log_file:
    csv_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n🔍 Evaluating: {file_name}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        header = [col.strip() for col in lines[0].strip().split(",")]
        rows = [line.strip().split(",") for line in lines[1:]]
        cleaned_rows = [row for row in rows if len(row) == len(header)]
        df = pd.DataFrame(cleaned_rows, columns=header)
        df.columns = df.columns.str.strip()

        total = 0
        correct = 0

        for i, row in df.iterrows():
            try:
                question = row["question"]
                ground_truth = row["AnswerKey"]
            except KeyError:
                print(f"⚠️ Skipping malformed row in {file_name}")
                continue

            total += 1
            write_prompt_to_file(question)
            run_socrates()
            response = read_model_output()
            predicted = extract_letter(response)

            all_results.append({
                "source_file": file_name,
                "question_number": i + 1,
                "predicted_answer": predicted,
                "ground_truth": ground_truth
            })

            # === Append to LOG FILE ===
            log_file.write(f"--- {file_name} | Question {i + 1} ---\n")
            log_file.write(response + "\n\n")

            if predicted == ground_truth:
                correct += 1

            print(f"#{i + 1}: Predicted {predicted} | Ground Truth {ground_truth} | {'✅' if predicted == ground_truth else '❌'}")

        accuracy = correct / total if total > 0 else 0.0
        summary_lines.append(f"{file_name}: {accuracy:.2%} ({correct}/{total})")

# === SAVE RESULTS ===
pd.DataFrame(all_results).to_csv(RESULT_CSV, index=False)
with open(SUMMARY_TXT, "w") as f:
    f.write("\n".join(summary_lines))

print(f"\n✅ Results saved to: {RESULT_CSV}")
print(f"📊 Summary saved to: {SUMMARY_TXT}")
print(f"🗒️  Logs saved to: {LOG_TXT}")


import csv
import subprocess

CSV_FILE = "problem_statements.csv"

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompt = row["prompt"]
        print(f"\n=== Running prompt: {prompt} ===\n")

        subprocess.run(
            ["python", "-m", "agentic_paper.academy_agent", prompt],
            check=True
        )

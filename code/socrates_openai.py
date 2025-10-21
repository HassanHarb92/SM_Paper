#!/usr/bin/env python3
"""
OpenAI version of your Socratic agent runner.

- Reads:
    prompt.txt           (system prompt)
    user_prompt.txt      (user query)
- Writes:
    output.txt           (full agent response)

Env:
    OPENAI_API_KEY must be set.

Model:
    Default "gpt-4o". To pin a dated snapshot, set MODEL="gpt-4o-YYYY-MM-DD" if available.
"""

from pathlib import Path
import os
import re
from openai import OpenAI

# ---------- config ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("SM_TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("SM_TOP_P", "1.0"))
MAX_TOKENS = int(os.getenv("SM_MAX_TOKENS", "2000"))

SYSTEM_PATH = Path("prompt.txt")
USER_PATH = Path("user_prompt.txt")
OUT_PATH = Path("output.txt")

# ---------- load inputs ----------
def load_text(p: Path, required=True, name="file"):
    try:
        return p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(f"Missing {name}: {p.resolve()}")
        return ""

SYSTEM_PROMPT = load_text(SYSTEM_PATH, True, "system prompt")
USER_PROMPT = load_text(USER_PATH, True, "user prompt")

# ---------- call OpenAI ----------
client = OpenAI()  # uses OPENAI_API_KEY

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    # If you constrain the final line, you can optionally add a stop sequence:
    # stop=["\n\nAnswer:"]
)

text = resp.choices[0].message.content or ""

# ---------- save output ----------
OUT_PATH.write_text(text, encoding="utf-8")

print("✅ Wrote response to", OUT_PATH.resolve())


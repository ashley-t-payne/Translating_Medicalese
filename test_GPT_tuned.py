#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: ashleypayne

------------------------------------------------------------
This script performs batch inference using a fine-tuned
GPT-2 medical language model.

Primary Workflow:
1. Reload a previously fine-tuned GPT-2 model
2. Load a CSV file containing medical terms
3. Convert each term into a structured medical question
4. Generate deterministic medical definitions using beam search
5. Save all generated outputs to a new CSV file

Use Case:
This script is used to generate structured medical test data
for evaluation, benchmarking, or downstream NLP pipelines.
------------------------------------------------------------
"""

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ============================
# CONFIGURATION SETTINGS
# ============================

# Directory where fine-tuned GPT-2 model is stored
SAVE_DIR = "/Users/ashleypayne/Desktop/cs385/gpt2-medical-finetuned"

# Input CSV containing test medical terms
INPUT_CSV = "medicalTermsStudySet_v1.csv"

# Output CSV where generated results will be saved
OUTPUT_CSV = "gpt2_generated_test_data.csv"

# Select Apple Silicon GPU (MPS) if available, otherwise use CPU
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ============================
# LOAD MODEL & TOKENIZER
# ============================

print("Reloading GPT-2 model...")

# Load fine-tuned model weights
model = GPT2LMHeadModel.from_pretrained(
    SAVE_DIR
).to(device)

# Load tokenizer associated with fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained(SAVE_DIR)

# Set padding token to EOS for proper generation behavior
tokenizer.pad_token = tokenizer.eos_token

# Switch model to evaluation mode (disables dropout)
model.eval()


# ============================
# LOAD CSV & KEEP LAST COLUMN
# ============================

# Load dataset into DataFrame
df = pd.read_csv(INPUT_CSV)

# Extract ONLY the last column as the list of medical terms
terms = df.iloc[:, -1].dropna().astype(str).tolist()

print(f"Loaded {len(terms)} test terms.")


# ============================
# QUESTION / INFERENCE FUNCTION
# ============================
"""
Generates a deterministic medical-style answer using the
fine-tuned GPT-2 model.

This function:
- Formats the question using Q&A structure
- Applies beam-search decoding for structured output
- Suppresses randomness for reproducible results
- Removes the original prompt from the final output

:param question: Input question string
                 (e.g., "What is Peristalsis?")
:param max_new_tokens: Maximum number of tokens to generate
:return: Generated medical answer as a clean string
"""

def ask(question: str, max_new_tokens: int = 40) -> str:

    # Construct structured medical prompt
    prompt = f"Question: {question}\nAnswer:"

    # Tokenize prompt and move tensors to selected device
    encoded = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    # Generate deterministic model output using beam search
    output = model.generate(
        **encoded,

        max_new_tokens=max_new_tokens,

        do_sample=False,        # Fully deterministic output
        num_beams=4,            # Enforces structured decoding
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,

        pad_token_id=tokenizer.eos_token_id
    )

    # Decode output into readable text
    decoded = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    # Remove original prompt from generated response
    answer = decoded.split("Answer:")[-1].strip()

    return answer


# ============================
# RUN GPT-2 ON ALL TERMS
# ============================

results = []

for i, term in enumerate(terms, start=1):
    print(f"[{i}/{len(terms)}] Generating: {term}")

    try:
        # Ask the model a structured medical question
        answer = ask(f"What is {term}?")

    except Exception as e:
        # Fail-safe to prevent a single error from breaking the run
        answer = f"ERROR: {e}"

    # Store results for CSV export
    results.append({
        "Term": term,
        "GPT2_Output": answer
    })


# ============================
# SAVE RESULTS TO NEW CSV
# ============================

# Convert output list to DataFrame
out_df = pd.DataFrame(results)

# Save without row index
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ GPT-2 TEST DATA SAVED TO: {OUTPUT_CSV}")

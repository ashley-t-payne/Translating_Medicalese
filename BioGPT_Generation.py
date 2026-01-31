#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Dec  7 00:58:19 2025

@author: ashleypayne

------------------------------------------------------------
This script performs batch biomedical definition generation
using Microsoft's BioGPT language model.

Primary Workflow:
1. Load a CSV file containing medical terms
2. Keep ONLY the final column as the term list
3. Convert each term into a BioGPT-friendly continuation prompt
4. Generate deterministic biomedical definitions
5. Save all generated outputs to a new CSV file

Use Case:
This script is designed for large-scale biomedical glossary
generation using controlled, deterministic BioGPT inference.
------------------------------------------------------------
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================
# CONFIGURATION SETTINGS
# ============================

# Input CSV containing medical terms
INPUT_CSV = "medicalTermsStudySet_v1.csv"

# Output CSV where generated BioGPT definitions will be saved
OUTPUT_CSV = "biogpt_generated_definitions.csv"

# Pretrained biomedical language model
MODEL_NAME = "microsoft/biogpt"


# ============================
# LOAD MODEL & TOKENIZER
# ============================

print("Loading BioGPT...")

# Load tokenizer associated with BioGPT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load BioGPT language model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Optional GPU acceleration for Apple Silicon
# Uncomment if running on compatible hardware
# device = torch.device("mps")
# model.to(device)


# ============================
# LOAD CSV & KEEP LAST COLUMN
# ============================

# Load dataset into a DataFrame
df = pd.read_csv(INPUT_CSV)

# Extract ONLY the last column as the list of medical terms
terms = df.iloc[:, -1].dropna().astype(str).tolist()

print(f"Loaded {len(terms)} medical terms.")


# ============================
# BIOGPT GENERATION FUNCTION
# ============================
"""
Generates a biomedical definition for a given term
using deterministic BioGPT sentence continuation.

This function:
- Converts the term into a BioGPT-friendly prompt
- Applies deterministic decoding
- Suppresses repetition for cleaner definitions
- Removes the repeated prompt from the final output

:param term: Medical term to generate a definition for
:return: Generated biomedical definition as a string
"""
def generate_biogpt_definition(term: str) -> str:

    # Construct BioGPT continuation-style prompt
    prompt = f"{term} is "

    # Tokenize prompt into model-ready tensors
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # If GPU is enabled, move tensors to device:
    # inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate deterministic biomedical definition
    outputs = model.generate(
        **inputs,

        max_new_tokens=60,        # Keeps output concise and definitional

        do_sample=False,         # Disables randomness
        repetition_penalty=1.25, # Discourages repeated phrasing
        no_repeat_ngram_size=3,  # Prevents local word duplication

        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode generated tokens into readable text
    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove the duplicated prompt from the front of the output
    definition = text.replace(
        prompt,
        ""
    ).strip()

    return definition


# ============================
# RUN BIOGPT ON ALL TERMS
# ============================

results = []

for i, term in enumerate(terms, start=1):
    print(f"[{i}/{len(terms)}] Generating: {term}")

    try:
        # Generate biomedical definition for current term
        definition = generate_biogpt_definition(term)

    except Exception as e:
        # Fail-safe: capture error instead of crashing pipeline
        definition = f"ERROR: {e}"

    # Store structured output for CSV export
    results.append({
        "Term": term,
        "BioGPT_Definition": definition
    })


# ============================
# SAVE RESULTS TO NEW CSV
# ============================

# Convert results list into DataFrame
out_df = pd.DataFrame(results)

# Save to CSV without index column
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n BIOGPT DEFINITIONS SAVED TO: {OUTPUT_CSV}")

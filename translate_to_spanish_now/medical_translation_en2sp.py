#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 21:17:44 2025

@author: ashleypayne
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Dec  9 14:22:00 2025

@author: ashleypayne

------------------------------------------------------------
This script performs batch English → Spanish medical
translation using a fine-tuned Marian-based medical
translation model from Hugging Face.

Model Used:
ayoubkirouane/Med_English2Spanish

Primary Workflow:
1. Load a CSV containing English medical definitions
2. Translate each definition into Spanish
3. Store both English and Spanish in a new CSV file

Use Case:
This script is used to create a bilingual medical dataset
for evaluation, benchmarking, or training multilingual models.
------------------------------------------------------------
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================
# CONFIGURATION SETTINGS
# ============================

# Input CSV containing English medical definitions
INPUT_CSV = "biogpt_generated_definitions.csv"

# Output CSV where Spanish translations will be stored
OUTPUT_CSV = "medical_english_to_spanish.csv"

# Medical translation model
MODEL_NAME = "ayoubkirouane/Med_English2Spanish"

# Select Apple Silicon GPU (MPS) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ============================
# LOAD MODEL & TOKENIZER
# ============================

print("Loading English → Spanish medical translation model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

model.eval()


# ============================
# LOAD ENGLISH DEFINITIONS CSV
# ============================

df = pd.read_csv(INPUT_CSV)

# ✅ Automatically select the column that contains English definitions
# Adjust the column name below if needed
ENGLISH_COLUMN = "BioGPT_Definition"

if ENGLISH_COLUMN not in df.columns:
    raise ValueError(
        f"Expected column '{ENGLISH_COLUMN}' not found in {INPUT_CSV}.\n"
        f"Available columns: {list(df.columns)}"
    )

english_texts = df[ENGLISH_COLUMN].dropna().astype(str).tolist()

print(f"Loaded {len(english_texts)} English medical definitions.")


# ============================
# TRANSLATION FUNCTION
# ============================

def translate_english_to_spanish(text: str, max_new_tokens: int = 80) -> str:
    """
    Translates an English medical definition into Spanish.

    This function:
    - Tokenizes the English input text
    - Runs Marian-based Seq2Seq medical translation
    - Decodes model output into readable Spanish

    :param text: English medical definition to translate
    :param max_new_tokens: Maximum number of tokens generated
    :return: Spanish translation as a string
    """

    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )

    spanish_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return spanish_text


# ============================
# RUN TRANSLATION ON ALL DEFINITIONS
# ============================

spanish_results = []

for i, english_text in enumerate(english_texts, start=1):
    print(f"[{i}/{len(english_texts)}] Translating...")

    try:
        spanish_text = translate_english_to_spanish(english_text)

    except Exception as e:
        spanish_text = f"ERROR: {e}"

    spanish_results.append(spanish_text)


# ============================
# SAVE RESULTS TO NEW CSV
# ============================

df["Spanish_Translation"] = spanish_results

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n English → Spanish medical translations saved to: {OUTPUT_CSV}")

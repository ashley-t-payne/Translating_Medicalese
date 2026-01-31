#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 21:51:41 2025

@author: ashleypayne
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERTScore Evaluation for Unsupervised Medical Translation
---------------------------------------------------------
Evaluates semantic similarity between:
- Original English
- Back-Translated English

Outputs:
- Precision
- Recall
- F1 Score
"""

import pandas as pd
from bert_score import score

# ============================
# CONFIG
# ============================

INPUT_CSV = "medical_english_to_spanish.csv"
REFERENCE_COL = "BioGPT_Definition"
PREDICTION_COL = "Spanish_Translation"

# ============================
# LOAD DATA
# ============================

df = pd.read_csv(INPUT_CSV)

references = df[REFERENCE_COL].astype(str).tolist()
predictions = df[PREDICTION_COL].astype(str).tolist()

print(f"Loaded {len(predictions)} translation pairs for evaluation.")

# ============================
# COMPUTE BERTSCORE
# ============================

P, R, F1 = score(
    predictions,
    references,
    lang="en",
    model_type="microsoft/deberta-xlarge-mnli"
)

# ============================
# RESULTS
# ============================

df["BERT_Precision"] = P.numpy()
df["BERT_Recall"] = R.numpy()
df["BERT_F1"] = F1.numpy()

print("\n BERTScore Evaluation Complete\n")
print(f"Average Precision: {df['BERT_Precision'].mean():.4f}")
print(f"Average Recall:    {df['BERT_Recall'].mean():.4f}")
print(f"Average F1 Score:  {df['BERT_F1'].mean():.4f}")

# ============================
# SAVE RESULTS
# ============================

OUTPUT_CSV = "bertscore_translation_eval.csv"
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n Saved evaluation to: {OUTPUT_CSV}")

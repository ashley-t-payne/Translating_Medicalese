#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Dec  7 00:31:01 2025

@author: ashleypayne

------------------------------------------------------------
This script generates a biomedical definition using the
FLAN-T5 sequence-to-sequence language model.

Primary Workflow:
1. Load the FLAN-T5 tokenizer and model
2. Provide an instruction-style medical prompt
3. Generate a concise, textbook-quality definition
4. Apply deterministic beam-search decoding
5. Print the generated medical output

Use Case:
This script is optimized for structured, instruction-following
medical definitions using encoder–decoder Transformers.
------------------------------------------------------------
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ============================
# MODEL CONFIGURATION
# ============================

# Pretrained instruction-following model
model_name = "google/flan-t5-base"


# ============================
# LOAD TOKENIZER & MODEL
# ============================

# Load tokenizer associated with FLAN-T5
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load FLAN-T5 sequence-to-sequence language model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# ============================
# INSTRUCTION-STYLE PROMPT SETUP
# ============================

# This structured instruction prompt forces:
# - Medical domain behavior
# - Short, factual definition style
# - Clear output constraints
text = (
    "You are a medical textbook.\n"
    "Define the anatomical structure called an adenoid.\n"
    "Give a short, clear definition."
)


# ============================
# TOKENIZATION
# ============================

# Convert instruction prompt into model-ready tensors
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True
)


# ============================
# TEXT GENERATION
# ============================

outputs = model.generate(
    **inputs,

    # Controls length of medical definition
    max_new_tokens=80,

    # Disable randomness for reproducible results
    do_sample=False,

    # Beam search improves factual grounding
    num_beams=5,

    # Stops generation once best sequence completes
    early_stopping=True,

    # Strong repetition control
    repetition_penalty=2.0,
    no_repeat_ngram_size=4,

    # Required padding behavior for Seq2Seq models
    pad_token_id=tokenizer.eos_token_id
)


# ============================
# OUTPUT DECODING
# ============================

# Convert generated token IDs into readable medical text
answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

# Print the generated biomedical definition
print(answer)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: ashleypayne

------------------------------------------------------------
This script performs dataset reduction and deduplication
on a BioGPT / GPT-style training dataset stored as plain text.

Each sample in the dataset follows this format:

Question: What is X?
Answer: ...
<|endoftext|>

Primary Workflow:
1. Load the full raw dataset from a text file
2. Split the dataset into individual Q&A blocks
3. Extract ONLY the answer portion from each block
4. Normalize each answer for safe duplicate detection
5. Remove duplicate answer entries
6. Save a reduced, deduplicated dataset to a new file

Use Case:
This script is used to eliminate repeated answers before
model fine-tuning in order to:
- Reduce training time
- Prevent overfitting on repeated definitions
- Improve dataset diversity
------------------------------------------------------------
"""

import re


# ============================
# CONFIGURATION SETTINGS
# ============================

# Input file containing full raw Q&A dataset
INPUT_FILE = "final_data_2.txt"

# Output file where the reduced dataset will be saved
OUTPUT_FILE = "final_data_2_reduced.txt"


# ============================
# INITIALIZE TRACKING STRUCTURES
# ============================

# Stores normalized answers for duplicate detection
seen_answers = set()

# Stores unique Q&A blocks that will be kept
kept_blocks = []


# ============================
# LOAD RAW DATASET
# ============================

# Read entire dataset into memory as a single string
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# Split dataset into individual Q&A blocks using EOS delimiter
blocks = content.split("<|endoftext|>")


# ============================
# DEDUPLICATION PIPELINE
# ============================

for block in blocks:
    # Remove leading and trailing whitespace
    block = block.strip()

    # Skip empty blocks
    if not block:
        continue

    # Extract ONLY the answer portion using regex
    match = re.search(
        r"Answer:\s*(.+)",
        block,
        re.DOTALL
    )

    # Skip malformed blocks that do not contain an answer
    if not match:
        continue

    # Extract raw answer text
    answer = match.group(1).strip()

    # Normalize answer for duplicate detection:
    # - Convert to lowercase
    # - Collapse all whitespace into single spaces
    norm_answer = re.sub(
        r"\s+",
        " ",
        answer.lower()
    )

    # If this answer has not been seen before, keep it
    if norm_answer not in seen_answers:
        seen_answers.add(norm_answer)
        kept_blocks.append(block)


# ============================
# DATASET REDUCTION REPORT
# ============================

print(f"Original samples: {len(blocks)}")
print(f"Reduced samples:  {len(kept_blocks)}")


# ============================
# SAVE REDUCED DATASET
# ============================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for block in kept_blocks:
        f.write(block.strip() + "\n<|endoftext|>\n\n")

print(f"\n Reduced dataset saved as: {OUTPUT_FILE}")

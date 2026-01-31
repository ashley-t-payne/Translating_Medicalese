#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Dec  3 10:18:35 2025

@author: ashleypayne

------------------------------------------------------------
This script evenly splits a large text file into multiple
smaller text files. 

Primary Workflow:
1. Load all non-empty lines from an input text file
2. Compute an even chunk size based on the number of splits
3. Divide the data into multiple segments
4. Write each segment to a separate output file

Use Case:
This script is commonly used to:
- Reduce large datasets for faster processing
- Partition vocabularies for parallel experiments
- Create smaller training subsets
------------------------------------------------------------
"""

# ============================
# CONFIGURATION SETTINGS
# ============================

# Input file containing the full dataset
INPUT_FILE = "wordlist.txt"

# Output files that will store split segments
OUTPUT_FILES = [
    "first.txt",
    "second.txt",
    "third.txt",
    "fourth.txt",
    "fifth.txt"
]

# Total number of desired output splits
NUM_SPLITS = 5


# ============================
# MAIN SPLITTING FUNCTION
# ============================

def main():
    """
    Splits the input text file into multiple smaller files.

    This function:
    - Reads all non-empty lines from the input file
    - Computes evenly sized chunks based on NUM_SPLITS
    - Writes each chunk to its corresponding output file
    - Ensures the final chunk captures any remaining lines

    :return: None
    """

    # ----------------------------
    # LOAD FULL DATASET
    # ----------------------------

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # Keep only non-empty lines
        lines = [
            line for line in f.readlines()
            if line.strip()
        ]

    total = len(lines)
    print(f"Loaded {total} lines from {INPUT_FILE}")


    # ----------------------------
    # COMPUTE CHUNK SIZE
    # ----------------------------

    # Integer division ensures near-even distribution
    chunk_size = total // NUM_SPLITS
    print(f"Each chunk will have about {chunk_size} lines")


    # ----------------------------
    # SPLIT AND WRITE TO FILES
    # ----------------------------

    for i in range(NUM_SPLITS):

        # Compute slice boundaries
        start = i * chunk_size

        # Ensure final chunk captures all remaining lines
        end = (i + 1) * chunk_size if i < NUM_SPLITS - 1 else total

        # Extract chunk of data
        chunk = lines[start:end]

        # Write chunk to corresponding output file
        with open(OUTPUT_FILES[i], "w", encoding="utf-8") as f:
            f.writelines(chunk)

        print(
            f"Wrote {len(chunk)} lines to {OUTPUT_FILES[i]} "
            f"({start} to {end})"
        )


# ============================
# SCRIPT ENTRY POINT
# ============================

if __name__ == "__main__":
    main()

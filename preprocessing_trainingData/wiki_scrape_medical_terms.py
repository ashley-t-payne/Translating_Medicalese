#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Dec  3 10:00:52 2025

@author: ashleypayne

------------------------------------------------------------
This script scrapes medical term definitions from Wikipedia
using the official REST API and formats the results for
GPT-style training.

Primary Workflow:
1. Load a list of medical terms from a text file
2. Fetch clean summary definitions from Wikipedia
3. Store raw results in a JSON file
4. Convert valid definitions into GPT-2 Q&A training format
5. Save formatted training samples to a text file

Use Case:
This script is designed to automatically build a structured,
high-quality medical training dataset for GPT-2 fine-tuning.
------------------------------------------------------------
"""

import requests
import json
from urllib.parse import quote


# ============================
# CONFIGURATION SETTINGS
# ============================

# Input file containing medical vocabulary terms
WORDLIST_FILE = "fifth.txt"

# Output file for raw scraped Wikipedia definitions (JSON)
OUTPUT_JSON = "medical_definitions.json"

# Output training file in GPT Q&A format
OUTPUT_TRAINING = "medical_training5.txt"

# Wikipedia REST API base endpoint
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"


# ============================
# LOAD TERMS
# ============================

# Load all non-empty medical terms from file
with open(WORDLIST_FILE, "r", encoding="utf-8") as f:
    terms = [
        line.strip() for line in f.readlines()
        if line.strip()
    ]

print(f"Loaded {len(terms)} terms.")


# ============================
# WIKIPEDIA SCRAPING FUNCTION
# ============================
"""
Fetches a clean summary definition for a given term
using the official Wikipedia REST API.

This function:
- Encodes the term for safe URL usage
- Handles HTTP errors
- Rejects disambiguation pages
- Extracts only the plain summary text

:param term: Medical term to fetch a definition for
:return:
    - Clean definition string if found
    - None if page is missing, ambiguous, or invalid
"""

def fetch_wikipedia_definition(term: str) -> str | None:

    # Construct URL-safe API endpoint for the term
    url = WIKI_API + quote(term)

    try:
        # Perform HTTP request using a safe user-agent
        r = requests.get(
            url,
            headers={"User-Agent": "GPT-Training-Scraper/1.0"}
        )

        # Abort if HTTP request fails
        if r.status_code != 200:
            return None

        # Parse JSON response
        data = r.json()

        # Reject disambiguation pages (non-definitive)
        if "type" in data and data["type"] == "disambiguation":
            return None

        # Extract the clean summary text
        if "extract" in data and data["extract"]:
            return data["extract"]

        return None

    except Exception as e:
        # Fail-safe network error handling
        print("Error fetching:", term, e)
        return None


# ============================
# SCRAPE ALL TERMS
# ============================

definitions = {}

for i, term in enumerate(terms):
    print(f"[{i+1}/{len(terms)}] Fetching: {term}")

    definition = fetch_wikipedia_definition(term)

    if definition:
        definitions[term] = definition
    else:
        # Preserve term with no definition for audit purposes
        definitions[term] = None


print(
    f"\nScraping complete. "
    f"{sum(1 for v in definitions.values() if v)} terms found."
)


# ============================
# SAVE RAW DEFINITIONS AS JSON
# ============================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        definitions,
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"Saved raw definitions to {OUTPUT_JSON}")


# ============================
# BUILD GPT-2 TRAINING FILE (Q&A FORMAT)
# ============================

with open(OUTPUT_TRAINING, "w", encoding="utf-8") as f:

    for term, definition in definitions.items():

        # Skip missing or invalid definitions
        if definition is None:
            continue

        # Structure data in GPT-style Q&A training format
        qna = (
            f"Question: What is {term}?\n"
            f"Answer: {definition}\n"
            "<|endoftext|>\n\n"
        )

        f.write(qna)

print(f"Saved training dataset to {OUTPUT_TRAINING}")

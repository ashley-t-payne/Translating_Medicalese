#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: ashleypayne

------------------------------------------------------------
This script implements a full Retrieval-Augmented Generation
(RAG) pipeline using:

- SentenceTransformers for semantic embedding and retrieval
- BioGPT for grounded biomedical text generation

Primary Workflow:
1. Load a CSV file of medical terms
2. Create semantic embeddings for all terms
3. Retrieve the most relevant reference term for each query
4. Inject the retrieved term into a BioGPT prompt
5. Generate a grounded biomedical definition
6. Save all results to a new CSV file

Use Case:
This system improves BioGPT output accuracy by grounding
generation in semantically similar glossary terms.
------------------------------------------------------------
"""

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util


# ============================
# CONFIGURATION SETTINGS
# ============================

# Input CSV containing raw medical terms
INPUT_CSV = "medicalTermsStudySet_v1.csv"

# Output CSV that will store RAG-enhanced BioGPT definitions
OUTPUT_CSV = "biogpt_rag_generated_definitions.csv"

# Sentence embedding model for semantic retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Biomedical text generation model
GENERATION_MODEL = "microsoft/biogpt"


# ============================
# LOAD MODELS
# ============================

print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Loading BioGPT...")
tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL)

# Optional GPU acceleration for Apple Silicon
# device = torch.device("mps")
# model.to(device)


# ============================
# LOAD CSV & KEEP LAST COLUMN
# ============================

# Load full dataset into DataFrame
df = pd.read_csv(INPUT_CSV)

# Extract ONLY the last column, drop missing values,
# convert everything to strings, then to a list
terms = df.iloc[:, -1].dropna().astype(str).tolist()

print(f"Loaded {len(terms)} medical terms.")


# ============================
# BUILD RETRIEVAL INDEX (EMBEDDINGS)
# ============================

print("Creating embeddings for RAG...")

# Convert all glossary terms into dense semantic vectors
corpus_embeddings = embedder.encode(
    terms,
    convert_to_tensor=True
)


# ============================
# RETRIEVAL FUNCTION
# ============================

def retrieve_context(query, top_k=1):
    """
    Retrieves the most semantically similar glossary term
    to the input query using cosine similarity.

    :param query: Input medical term to retrieve context for
    :param top_k: Number of top matches to return (default = 1)
    :return: Most semantically similar medical reference term
    """

    # Generate embedding for the query term
    query_embedding = embedder.encode(
        query,
        convert_to_tensor=True
    )

    # Compute cosine similarity with entire corpus
    scores = util.cos_sim(
        query_embedding,
        corpus_embeddings
    )[0]

    # Get index of highest similarity score
    best_idx = torch.argmax(scores).item()

    return terms[best_idx]


# ============================
# BIOGPT + RAG GENERATION FUNCTION
# ============================

def generate_biogpt_rag_definition(term):
    """
    Generates a grounded biomedical definition using RAG.

    Pipeline:
    1. Retrieve semantically closest glossary term
    2. Inject retrieved term into BioGPT prompt
    3. Generate a grounded biomedical definition

    :param term: Medical term to define
    :return:
        - Generated biomedical definition (string)
        - Retrieved reference context (string)
    """

    # Step 1: Retrieve closest reference term
    retrieved_context = retrieve_context(term)

    # Step 2: Construct grounded biomedical prompt
    prompt = (
        f"Medical reference term: {retrieved_context}. "
        f"{term} is "
    )

    # Tokenize prompt for BioGPT
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # Step 3: Generate grounded biomedical definition
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        repetition_penalty=1.25,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode raw output text
    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove prompt from generated text
    definition = text.replace(
        prompt,
        ""
    ).strip()

    return definition, retrieved_context


# ============================
# RUN RAG + BIOGPT ON ALL TERMS
# ============================

results = []

for i, term in enumerate(terms, start=1):
    print(f"[{i}/{len(terms)}] RAG Generating: {term}")

    try:
        definition, context = generate_biogpt_rag_definition(term)

    except Exception as e:
        # Fail-safe to prevent pipeline crash
        definition = f"ERROR: {e}"
        context = "N/A"

    # Store structured output for CSV export
    results.append({
        "Term": term,
        "Retrieved_Context": context,
        "BioGPT_RAG_Definition": definition
    })


# ============================
# SAVE RESULTS TO NEW CSV
# ============================

# Convert list of results into DataFrame
out_df = pd.DataFrame(results)

# Save to CSV without index column
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nRAG + BioGPT DEFINITIONS SAVED TO: {OUTPUT_CSV}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashleypayne


------------------------------------------------------------
This script fine-tunes a GPT-2 language model on a custom
medical question-answer (Q&A) glossary dataset stored in a
plain text file.

Expected input format in final_data_2_reduced.txt:

Question: What is 11-dehydrocorticosterone?
Answer: 11-Dehydrocorticosterone (11-DHC) ...
<|endoftext|>

Question: What is Adipose?
Answer: ...
<|endoftext|>

------------------------------------------------------------
Primary Goals:
- Load raw Q&A data from a text file
- Tokenize and convert data into a PyTorch dataset
- Fine-tune GPT-2 using Hugging Face Trainer
- Save the trained model
- Test the model with medical-style questions
------------------------------------------------------------
"""

import os
import random
import torch
import numpy as np
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


# ==============================
# DEVICE SETUP (APPLE SILICON SAFE)
# ==============================

# Select Apple MPS GPU if available, otherwise fall back to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using device: CPU")


# ==============================
# REPRODUCIBILITY
# ==============================

# Set fixed random seed so training behavior is repeatable
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Only applies if CUDA is available (not used on M1, but kept for portability)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==============================
# LOAD RAW Q&A TEXT DATA
# ==============================

DATA_FILE = "final_data_2_reduced.txt"

# Ensure training file exists before attempting to read it
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in current directory.")

# Load entire dataset as a single string
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split on <|endoftext|> so each chunk is one full Q&A entry
examples = []

for chunk in raw_text.split("<|endoftext|>"):
    chunk = chunk.strip()

    if not chunk:
        continue

    # Ensure each example explicitly ends with the EOS token
    examples.append(chunk + "\n<|endoftext|>")

print(f"Loaded {len(examples)} Q&A training examples.")

# Randomize example order to improve training generalization
random.shuffle(examples)


# ==============================
# LOAD GPT-2 MODEL & TOKENIZER
# ==============================

# Model selection (can be scaled up on stronger GPUs)
model_name = "gpt2"

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Use EOS token as padding token (required for GPT-2 training)
tokenizer.pad_token = tokenizer.eos_token

# Load pretrained GPT-2 language model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Resize embeddings in case tokenizer size changed
model.resize_token_embeddings(len(tokenizer))

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

# Move model to selected device
model.to(device)


# ==============================
# TOKENIZATION
# ==============================

MAX_LEN = 128  # Increase to 256–512 if definitions are long

def tokenize_function(batch_texts):
    """
    Tokenizes a list of training texts for GPT-2.

    This function applies:
    - Truncation to MAX_LEN
    - Attention mask creation
    - No padding (handled later by data collator)

    :param batch_texts: List of Q&A training strings
    :return: Dictionary containing tokenized input IDs and attention masks
    """
    return tokenizer(
        batch_texts,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_attention_mask=True
    )

# Tokenize entire dataset
tokenized = tokenize_function(examples)


# ==============================
# CUSTOM PYTORCH DATASET
# ==============================

class QADataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for medical Q&A language model fine-tuning.

    This dataset returns:
    - input_ids tensor
    - attention_mask tensor

    Each item corresponds to one Q&A example.
    """

    def __init__(self, tokenized_batch):
        """
        Initializes the dataset using tokenized inputs.

        :param tokenized_batch: Output dictionary from tokenizer
        """
        self.input_ids = tokenized_batch["input_ids"]
        self.attention_mask = tokenized_batch["attention_mask"]

    def __len__(self):
        """
        Returns total number of training examples.

        :return: Integer dataset size
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Returns one training example as PyTorch tensors.

        :param idx: Index of the sample to retrieve
        :return: Dictionary with input_ids and attention_mask tensors
        """
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }


# Instantiate dataset
dataset = QADataset(tokenized)


# ==============================
# DATA COLLATOR (DYNAMIC PADDING)
# ==============================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal language modeling (NOT masked)
)


# ==============================
# TRAINING CONFIGURATION (M1 FRIENDLY)
# ==============================

training_args = TrainingArguments(
    output_dir="gpt2-medical-finetuned",
    use_cpu=True,                        # Forces CPU-safe behavior on M1
    overwrite_output_dir=True,

    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,     # Effective batch size = 4
    gradient_checkpointing=True,

    learning_rate=1e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.03,

    logging_steps=10,
    save_steps=200,
    save_total_limit=2,

    report_to="none",
    fp16=False,
    bf16=False,
)


# ==============================
# TRAINER INITIALIZATION
# ==============================

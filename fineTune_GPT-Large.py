#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 00:10:50 2025

@author: ashleypayne
"""

import pandas as pd
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

"""
DEVICE SETUP (M1-SAFE)
"""
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using device: CPU")


"""
LOAD AND PREPROCESS DATA
Convert glossary rows into Q&A format:

Question: What is <Word>?
Answer: <Meaning>

This teaches GPT-2 the exact behavior we want.
"""
data = pd.read_csv("final_data.txt").fillna("")

# If column exists, drop it
if "Example Term" in data.columns:
    data.drop("Example Term", axis=1, inplace=True)

questions = []
for _, row in data.iterrows():
    word = str(row["Word"])
    meaning = str(row["Meaning"])

    q = f"Question: What is {word}?\nAnswer: {meaning}\n<|endoftext|>"
    questions.append(q)

print(f"Created {len(questions)} Q&A training records.")


"""
LOAD SMALL GPT-2 MODEL (SAFE FOR M1)
"""
model_name = "gpt2"  # Not gpt2-large!!! Too big.

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()
model.to(device)


"""
TOKENIZATION
Short sequence length because Q&A pairs are small.
"""
MAX_LEN = 128

encodings = tokenizer(
    questions,
    truncation=True,
    max_length=MAX_LEN,
    padding="max_length",
    return_tensors="pt",
)

encodings["labels"] = encodings["input_ids"].clone()


class QADataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return self.enc["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}


dataset = QADataset(encodings)


"""
TRAINING CONFIG — SAFE FOR M1
ONE EPOCH ONLY (prevent overfitting)
M1 Experienced RAM error when using GPU instead of CPU
"""
training_args = TrainingArguments(
    output_dir="gpt2-medical-finetuned",
    metric_for_best_model="loss",
    greater_is_better=False,
    overwrite_output_dir=True,
    num_train_epochs=3,                 # do NOT use more or you overfit
    per_device_train_batch_size=1,      # keeps memory low
    gradient_accumulation_steps=4,
    learning_rate=5e-5,                 #look into auto adjusted learning rate
    #warmup_steps=100,
    label_smoothing_factor = 0.1,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("\nStarting training...\n")
trainer.train()
print("\nTraining complete.\n")


"""
SAVE & RELOAD
"""
save_dir = "gpt2-medical-finetuned"
print(f"Saving model to {save_dir}...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Reloading model...")
model = GPT2LMHeadModel.from_pretrained(save_dir).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(save_dir)


"""
TEST THE MODEL
"""
def ask(question):
    prompt = f"Question: {question}\nAnswer:"
    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **encoded,
        max_new_tokens=40,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.95,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


print("\n--- TEST OUTPUTS ---\n")
print(ask("What is Adipose?"))
print()

'''
print(ask("What is Peristalsis?"))
print()
print(ask("What is a Neuron?"))
'''
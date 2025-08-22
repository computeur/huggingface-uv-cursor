#!/usr/bin/env python3
"""
Simple training example using Accelerate with MPS support.
This demonstrates how to set up a basic training loop with Hugging Face.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from accelerate import Accelerator


def main():
    # Initialize accelerator
    accelerator = Accelerator()

    # Check device
    device = accelerator.device
    print(f"Using device: {device}")

    # Use a tiny model for a fast demo
    model_name = "sshleifer/tiny-gpt2"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load a tiny slice of a dataset for demo
    print("Loading dataset...")
    dataset = load_dataset("squad", split="train[:100]")

    # Simple preprocessing function
    def preprocess_function(examples):
        questions = examples["question"]
        contexts = examples["context"]
        texts = [f"Question: {q} Context: {c}" for q, c in zip(questions, contexts)]

        # Important: don't return torch tensors from map() – return lists
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        # For demo purposes, use inputs as labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Preprocess dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Tell datasets to return torch tensors
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Create DataLoader
    train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Prepare with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    print("Starting training (demo: 5 steps)...")
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step >= 5:  # Just run a few steps for demo
            break
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        if step % 1 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    print("Training demo completed!")


if __name__ == "__main__":
    main()

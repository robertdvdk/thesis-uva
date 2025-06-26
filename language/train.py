# train.py

"""
Main training script for the Transformer model.

This script orchestrates the entire process:
1. Imports configuration, data pipeline, and model from other files.
2. Initializes the model, optimizer, loss function, and learning rate scheduler.
3. Runs the main training loop for a fixed number of steps.
4. Periodically runs evaluation and saves the best model.
5. Saves the final model, logs, and configuration.
"""

import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple
import os
import math
from datetime import datetime
import json

# --- MODIFIED: Import Tokenizer for type hinting ---
from tokenizers import Tokenizer

# --- Import from our project files ---
from config import CONFIG
from data import DataPipeline
from model import Seq2SeqTransformer

# --- Global step counter for the training loop ---
g_step = 0


# --- Logger Class (no changes) ---
class Logger:
    """
    A simple logger that writes messages to a file and prints to the console.
    """

    def __init__(self, log_path: str):
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def log(self, message: str):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# --- calculate_accuracy (no changes) ---
def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """Calculates the token-level accuracy."""
    predictions = logits.argmax(dim=-1)
    non_padding_mask = targets != ignore_index
    correct = (predictions == targets) & non_padding_mask
    accuracy = correct.sum().item() / non_padding_mask.sum().item()
    return accuracy


# --- rate (no changes) ---
def rate(step: int, d_model: int, factor: float, warmup: int) -> float:
    """
    Learning rate schedule as defined in the "Attention Is All You Need" paper.
    (Noam Scheduler)
    """
    if step == 0:
        step = 1
    return factor * (
            d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# --- MODIFIED: `evaluate` now accepts a single tokenizer object ---
def evaluate(model: nn.Module, criterion: nn.Module,
             val_loader: torch.utils.data.DataLoader,
             device: torch.device,
             tokenizer: Tokenizer) -> Tuple[float, float, float]:  # MODIFIED: `tokenizer: Tokenizer`
    """Runs one full evaluation cycle."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    # MODIFIED: Get pad_token_id from the single tokenizer
    pad_token_id = tokenizer.token_to_id("[PAD]")

    with torch.no_grad():
        for batch in val_loader:
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            src_padding_mask = batch['attention_mask'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_padding_mask = (tgt_input != criterion.ignore_index)
            tgt_input_safe = tgt_input.clone()
            tgt_input_safe[tgt_input_safe == -100] = pad_token_id
            logits = model(src, tgt_input_safe, src_padding_mask, tgt_padding_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(logits, tgt_out, ignore_index=-100)

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader)
    avg_perplexity = math.exp(avg_loss)
    return avg_loss, avg_accuracy, avg_perplexity


# --- count_parameters (no changes) ---
def count_parameters(model: nn.Module, logger: Logger):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"\n--- Total trainable parameters: {total_params:,} ---")


def main():
    """Main execution function."""
    global g_step
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = CONFIG.get("run_name")
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join("checkpoints", run_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    log_path = os.path.join(checkpoint_dir, "training_log.txt")
    logger = Logger(log_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=4)

    logger.log(f"--- Configuration saved to: {config_path} ---")
    logger.log(f"--- Using device: {device} ---")
    logger.log(f"--- Checkpoints and logs will be saved in: {checkpoint_dir} ---")

    logger.log("\n--- Initializing Data Pipeline ---")
    pipeline = DataPipeline(config=CONFIG)
    # --- MODIFIED: Unpack the single tokenizer object ---
    train_loader, val_loader, tokenizer = pipeline.get_dataloaders()

    # --- MODIFIED: Use the single shared tokenizer for vocab size and pad token ---
    # The vocabulary is shared, so there's only one size.
    vocab_size = tokenizer.get_vocab_size()
    src_vocab_size = vocab_size
    tgt_vocab_size = vocab_size
    # There is only one PAD token ID in a shared vocabulary
    pad_token_id = tokenizer.token_to_id("[PAD]")

    logger.log("\n--- Initializing Model ---")
    model = Seq2SeqTransformer(
        num_encoder_layers=CONFIG["num_encoder_layers"],
        num_decoder_layers=CONFIG["num_decoder_layers"],
        d_model=CONFIG["d_model"], nhead=CONFIG["nhead"],
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=CONFIG["dim_feedforward"], dropout=CONFIG["dropout"]
    ).to(device)

    count_parameters(model, logger)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, CONFIG["d_model"], factor=CONFIG["lr_multiplier"],
                                    warmup=CONFIG["warmup_steps"]),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=CONFIG["label_smoothing"]
    )

    best_val_loss = float('inf')
    if CONFIG.get("load_model_path"):
        load_path = CONFIG["load_model_path"]
        if os.path.exists(load_path):
            logger.log(f"--- Loading checkpoint: {load_path} ---")
            checkpoint = torch.load(load_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('loss', float('inf'))
            if 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.log("--- Loaded scheduler state ---")
            if 'g_step' in checkpoint:
                g_step = checkpoint['g_step']
                logger.log(f"--- Resuming from global step: {g_step} ---")
            else:
                logger.log("--- WARNING: 'g_step' not found in checkpoint. Starting from step 0. ---")
        else:
            logger.log(f"--- WARNING: Checkpoint path not found: {load_path} ---")

    logger.log("\n--- Starting Training ---")

    max_steps = CONFIG["max_steps"]
    eval_interval = CONFIG["eval_interval"]

    model.train()  # Set model to training mode once

    while g_step < max_steps:
        for batch in train_loader:
            if g_step >= max_steps:
                break

            start_time_step = time.time()

            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            src_padding_mask = batch['attention_mask'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_padding_mask = (tgt_input != criterion.ignore_index)
            tgt_input_safe = tgt_input.clone()
            # --- MODIFIED: Use the new `pad_token_id` variable ---
            tgt_input_safe[tgt_input_safe == -100] = pad_token_id

            logits = model(src, tgt_input_safe, src_padding_mask, tgt_padding_mask)

            optimizer.zero_grad()
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["max_grad_norm"])
            optimizer.step()
            lr_scheduler.step()

            g_step += 1

            if g_step > 0 and g_step % eval_interval == 0:
                step_time = time.time() - start_time_step
                current_lr = lr_scheduler.get_last_lr()[0]

                logger.log("-" * 50)
                logger.log(f"Step: {g_step}/{max_steps} | LR: {current_lr:.6f} | "
                           f"Step Time: {step_time:.2f}s | Train Loss: {loss.item():.3f}")

                # --- MODIFIED: Pass the single tokenizer to `evaluate` ---
                val_loss, val_acc, val_ppl = evaluate(
                    model, criterion, val_loader, device, tokenizer
                )

                logger.log(
                    f"\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}% |  Val. PPL: {val_ppl:7.3f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                    torch.save({
                        'g_step': g_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'loss': val_loss,
                    }, best_model_path)
                    logger.log(f"\t -> Checkpoint saved to {best_model_path}")

                model.train()
                logger.log("-" * 50)

    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save({
        'g_step': g_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': best_val_loss,
    }, final_model_path)
    logger.log(f"\n--- Training Finished at step {g_step}. Final model saved to {final_model_path} ---")

    logger.close()


if __name__ == "__main__":
    main()
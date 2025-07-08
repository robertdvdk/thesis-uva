# train.py

"""
Main training script for the Transformer model.

This script orchestrates the entire process:
1. Imports configuration, data pipeline, and model from other files.
2. Initializes the model, optimizer, loss function, and learning rate scheduler.
3. Runs the main training loop for a fixed number of steps.
4. Periodically runs evaluation and saves the best model.
5. Implements early stopping if validation loss stagnates.
6. Saves the final model, logs, and configuration.
"""

import random
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple
import os
import math
from datetime import datetime
import json
import argparse
import sys
from tokenizers import Tokenizer
import torch.profiler

from config import CONFIG
from data import DataPipeline
from model import Seq2SeqTransformer
from hmodel import HSeq2SeqTransformer
import numpy as np
from hypll.manifolds.hyperboloid import Hyperboloid, Curvature
import hypll.optim

# Global step counter
g_step = 0


class Logger:
    """A simple logger that writes messages to a file and prints to the console."""

    def __init__(self, log_path: str):
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def log(self, message: str):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> Tuple[int, int]:
    """
    Calculates the total number of correct predictions and the total number of non-ignored tokens.
    Returns a tuple of (total_correct, total_tokens).
    """
    predictions = logits.argmax(dim=-1)
    non_padding_mask = targets != ignore_index
    total_tokens = non_padding_mask.sum().item()
    if total_tokens == 0:
        return 0, 0  # Avoid division by zero
    correct = (predictions == targets) & non_padding_mask
    total_correct = correct.sum().item()
    return total_correct, total_tokens


def rate(step: int, d_model: int, factor: float, warmup: int) -> float:
    """Learning rate schedule as defined in the "Attention Is All You Need" paper."""
    if step == 0:
        step = 1
    return factor * (
            d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def evaluate(model: nn.Module, criterion: nn.Module,
             val_loader: torch.utils.data.DataLoader,
             device: torch.device,
             tokenizer: Tokenizer) -> Tuple[float, float, float]:
    """Runs one full evaluation cycle."""
    model.eval()
    total_loss = 0
    total_correct_predictions = 0
    total_token_count = 0
    pad_token_id = tokenizer.token_to_id("[PAD]")

    with torch.no_grad():
        for batch in val_loader:
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            src_padding_mask = batch['attention_mask'].to(device)

            # --- STANDARD TRANSLATION SETUP ---
            tgt_input = tgt[:, :-1].clone()
            tgt_out = tgt[:, 1:].clone()

            # --- FIX: Sanitize the decoder input ---
            # Replace any -100 values with the padding token ID before passing to the model.
            # The model should not see the ignore_index.
            tgt_input[tgt_input == criterion.ignore_index] = pad_token_id

            tgt_padding_mask = (tgt_input != pad_token_id)
            tgt_out[tgt[:, 1:] == pad_token_id] = criterion.ignore_index

            logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask)
            # --- END STANDARD SETUP ---

            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()

            correct, total_tokens = calculate_accuracy(logits, tgt_out, ignore_index=criterion.ignore_index)
            total_correct_predictions += correct
            total_token_count += total_tokens

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_accuracy = total_correct_predictions / total_token_count if total_token_count > 0 else 0
    avg_perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, avg_accuracy, avg_perplexity


def count_parameters(model: nn.Module, logger: Logger):
    """Logs the number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"\n--- Total trainable parameters: {total_params:,} ---")


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main execution function."""
    global g_step

    parser = argparse.ArgumentParser(description="Train a Hyperbolic Transformer model.")
    parser.add_argument('--run_name', type=str, default=CONFIG['run_name'], help="A name for this training run.")
    parser.add_argument('--manifold', type=str, default=CONFIG['manifold'], help="Manifold of model.")
    parser.add_argument('--dropout', type=float, default=CONFIG['dropout'], help="Dropout parameter.")
    parser.add_argument('--load_model_path', type=str, default=CONFIG['load_model_path'], help="Which model to load.")
    parser.add_argument('--nhead', type=int, default=CONFIG['nhead'], help="Number of attention heads.")
    parser.add_argument('--d_model', type=int, default=CONFIG['d_model'], help="Model dimension.")
    parser.add_argument('--dim_feedforward', type=int, default=CONFIG['dim_feedforward'], help="MLP dimension.")
    parser.add_argument('--tokens_per_batch', type=int, default=CONFIG['tokens_per_batch'], help="tokens per batch.")
    args = parser.parse_args()
    CONFIG.update(vars(args))

    seed = CONFIG.get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert CONFIG["manifold"] in ["hyperboloid", "Euclidean"]
    if CONFIG["manifold"] == "hyperboloid":
        manifold = Hyperboloid(Curvature(value=np.log(np.exp(1) - 1)))

    run_name = CONFIG.get("run_name")
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(checkpoint_dir, "training_log.txt")
    logger = Logger(log_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=4)

    logger.log(f"--- Using Seed: {seed} ---")
    logger.log(f"--- Configuration saved to: {config_path} ---")
    logger.log(f"--- Using device: {device} ---")
    logger.log(f"--- Checkpoints and logs will be saved in: {checkpoint_dir} ---")

    logger.log("\n--- Initializing Data Pipeline ---")
    pipeline = DataPipeline(config=CONFIG)
    train_loader, val_loader, tokenizer = pipeline.get_dataloaders()

    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    logger.log(f"--- Vocabulary Size (including special tokens): {vocab_size} ---")

    pad_token_id = tokenizer.token_to_id("[PAD]")

    logger.log("\n--- Initializing Model ---")
    if CONFIG["manifold"] == "hyperboloid":
        model = HSeq2SeqTransformer(
            num_encoder_layers=CONFIG["num_encoder_layers"],
            num_decoder_layers=CONFIG["num_decoder_layers"],
            d_model=CONFIG["d_model"], nhead=CONFIG["nhead"],
            src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
            dim_feedforward=CONFIG["dim_feedforward"], dropout=CONFIG["dropout"],
            manifold=manifold,
        ).to(device)
    else:
        model = Seq2SeqTransformer(
            num_encoder_layers=CONFIG["num_encoder_layers"],
            num_decoder_layers=CONFIG["num_decoder_layers"],
            d_model=CONFIG["d_model"], nhead=CONFIG["nhead"],
            src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
            dim_feedforward=CONFIG["dim_feedforward"], dropout=CONFIG["dropout"],
        ).to(device)

    count_parameters(model, logger)
    optimizer = hypll.optim.RiemannianAdam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-8)

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
            checkpoint = torch.load(load_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('loss', float('inf'))
            if 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'g_step' in checkpoint:
                g_step = checkpoint['g_step']
        else:
            logger.log(f"--- WARNING: Checkpoint path not found: {load_path} ---")

    logger.log("\n--- Starting Training ---")

    max_steps = CONFIG["max_steps"]
    eval_interval = CONFIG["eval_interval"]
    model.train()
    early_stopping_patience = 5
    epochs_without_improvement = 0
    training_complete = False
    param_to_name = {param: name for name, param in model.named_parameters()}
    start_interval_time = time.time()

    while g_step < max_steps and not training_complete:
        for batch in train_loader:
            if g_step >= max_steps:
                training_complete = True
                break

            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            src_padding_mask = batch['attention_mask'].to(device)

            # --- STANDARD TRANSLATION SETUP ---
            tgt_input = tgt[:, :-1].clone()
            tgt_out = tgt[:, 1:].clone()

            # --- FIX: Sanitize the decoder input ---
            # Replace any -100 values with the padding token ID before passing to the model.
            tgt_input[tgt_input == criterion.ignore_index] = pad_token_id

            tgt_padding_mask = (tgt_input != pad_token_id)
            tgt_out[tgt[:, 1:] == pad_token_id] = criterion.ignore_index

            logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask)
            # --- END STANDARD SETUP ---

            optimizer.zero_grad()
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["max_grad_norm"])

            optimizer.step(param_to_name)
            lr_scheduler.step()
            g_step += 1

            if g_step > 0 and g_step % eval_interval == 0:
                interval_time = time.time() - start_interval_time
                current_lr = lr_scheduler.get_last_lr()[0]

                logger.log("-" * 50)
                logger.log(f"Step: {g_step}/{max_steps} | LR: {current_lr:.6f} | "
                           f"Interval Time: {interval_time:.2f}s | Train Loss: {loss.item():.3f}")
                start_interval_time = time.time()

                val_loss, val_acc, val_ppl = evaluate(
                    model, criterion, val_loader, device, tokenizer
                )

                logger.log(
                    f"\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}% |  Val. PPL: {val_ppl:7.3f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_model_path = os.path.join(checkpoint_dir, f"best_model.pth")
                    torch.save({
                        'g_step': g_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'loss': val_loss,
                    }, best_model_path)
                    logger.log(f"\t -> New best model saved to {best_model_path}")
                else:
                    epochs_without_improvement += 1
                    logger.log(
                        f"\t -> No improvement in validation loss for {epochs_without_improvement} consecutive evaluation(s).")

                if epochs_without_improvement >= early_stopping_patience:
                    logger.log(
                        f"\n--- Early stopping triggered after {epochs_without_improvement} evaluations without improvement. ---")
                    training_complete = True
                    break

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

"""
Optimized evaluation script for the Transformer model.

This script performs the following steps:
1. Loads a trained model checkpoint and its associated configuration.
2. Initializes the data pipeline to get the validation set and tokenizer.
3. Iterates through the validation dataset, processing an entire batch at a time.
4. For each batch, it generates translations using an efficient, batched greedy decoding method.
5. It compares the generated translations (candidates) against the ground truth (references).
6. It prints sample translations from the first batch for qualitative inspection.
7. It calculates and prints the overall BLEU score for the entire validation set.
"""

import torch
import torch.nn as nn
from torchmetrics.text import BLEUScore
import sys
import os
import json
from tokenizers import Tokenizer
import numpy as np
from tqdm import tqdm

from model import Seq2SeqTransformer
from hmodel import HSeq2SeqTransformer
from hypll.manifolds.hyperboloid import Hyperboloid, Curvature
from data import DataPipeline
from config import CONFIG as DEFAULT_CONFIG


def calculate_bleu(model: nn.Module,
                   val_loader: torch.utils.data.DataLoader,
                   tokenizer: Tokenizer,
                   device: torch.device,
                   max_len: int) -> float:
    """
    Calculates the BLEU score using efficient batched decoding
    and prints samples from every batch.
    """
    model.eval()
    candidates = []
    references = []
    bleu_metric = BLEUScore(n_gram=4)
    pad_token_id = tokenizer.token_to_id("[PAD]")

    print("\n--- Calculating BLEU score on the validation set ---")
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating BLEU")):
        src_batch = batch['input_ids'].to(device)
        src_mask_batch = batch['attention_mask'].to(device)
        tgt_batch = batch['labels']

        # ====================================================================
        #            🚀 OPTIMIZED BATCHED GREEDY DECODING 🚀
        # ====================================================================
        with torch.no_grad():
            batch_size = src_batch.size(0)
            sos_token_id = tokenizer.token_to_id("[SOS]")
            eos_token_id = tokenizer.token_to_id("[EOS]")

            memory = model.encode(src_batch, src_padding_mask=src_mask_batch)
            ys = torch.ones(batch_size, 1).fill_(sos_token_id).type(torch.long).to(device)
            finished_seq = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len - 1):
                tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device)
                out = model.decode(ys, memory, tgt_mask)

                last_token_output = out[:, -1]
                if hasattr(last_token_output, 'tensor'):
                    last_token_output = last_token_output.tensor

                prob = model.generator(last_token_output)
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.masked_fill(finished_seq, pad_token_id)
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                finished_seq |= (next_word == eos_token_id)

                if finished_seq.all():
                    break
        # =================== END BATCHED DECODING ===================

        # Decode candidates and prepare references for the current batch
        candidate_strs = tokenizer.decode_batch(ys.tolist(), skip_special_tokens=True)
        candidates.extend(candidate_strs)

        current_batch_references = []
        for i in range(tgt_batch.size(0)):
            ref_ids = [tid.item() for tid in tgt_batch[i] if tid != -100]
            ref_sentence = tokenizer.decode(ref_ids, skip_special_tokens=True)
            current_batch_references.append([ref_sentence])
        references.extend(current_batch_references)

        # # --- 🔍 Print sample translations from the current batch ---
        # print(f"\n--- Sample Translations (Batch {batch_idx + 1}/{len(val_loader)}) ---")
        # num_samples_to_print = min(3, src_batch.size(0))  # Print up to 3 samples
        # src_strs = tokenizer.decode_batch(src_batch[:num_samples_to_print].tolist(), skip_special_tokens=True)
        #
        # for i in range(num_samples_to_print):
        #     print(f"\n  SAMPLE {i + 1}")
        #     print(f"    SOURCE:    {src_strs[i]}")
        #     print(f"    REFERENCE: {current_batch_references[i][0]}")
        #     print(f"    GENERATED: {candidate_strs[i]}")
        # print("-" * 40)

    score = bleu_metric(candidates, references).item() * 100
    return score

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_model.pth>")
        sys.exit(1)

    model_path = sys.argv[1]
    checkpoint_dir = os.path.dirname(model_path)
    config_path = os.path.join(checkpoint_dir, 'config.json')

    if not os.path.exists(config_path):
        print(f"Error: config.json not found in checkpoint directory: {checkpoint_dir}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)
    print(f"--- Configuration loaded from: {config_path} ---")

    DEFAULT_CONFIG.update(CONFIG)
    DEFAULT_CONFIG['data_dir_raw'] = 'iwslt14_data_raw'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    pipeline = DataPipeline(config=DEFAULT_CONFIG)
    _, val_loader, tokenizer = pipeline.get_dataloaders()
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    print(f"--- Vocabulary Size: {vocab_size} ---")

    print(f"--- Initializing model on {DEFAULT_CONFIG['manifold']} manifold ---")
    if DEFAULT_CONFIG["manifold"] == "hyperboloid":
        manifold = Hyperboloid(Curvature(value=np.log(np.exp(1) - 1)))
        model = HSeq2SeqTransformer(
            num_encoder_layers=DEFAULT_CONFIG["num_encoder_layers"],
            num_decoder_layers=DEFAULT_CONFIG["num_decoder_layers"],
            d_model=DEFAULT_CONFIG["d_model"], nhead=DEFAULT_CONFIG["nhead"],
            src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
            dim_feedforward=DEFAULT_CONFIG["dim_feedforward"], dropout=DEFAULT_CONFIG["dropout"],
            manifold=manifold, impl=DEFAULT_CONFIG["impl"],
        ).to(device)
    else:
        model = Seq2SeqTransformer(
            num_encoder_layers=DEFAULT_CONFIG["num_encoder_layers"],
            num_decoder_layers=DEFAULT_CONFIG["num_decoder_layers"],
            d_model=DEFAULT_CONFIG["d_model"], nhead=DEFAULT_CONFIG["nhead"],
            src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
            dim_feedforward=DEFAULT_CONFIG["dim_feedforward"], dropout=DEFAULT_CONFIG["dropout"],
        ).to(device)

    print(f"--- Loading model weights from: {model_path} ---")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    max_len = DEFAULT_CONFIG.get('max_seq_len', 100)
    bleu_score = calculate_bleu(model, val_loader, tokenizer, device, max_len=max_len)

    print("-" * 50)
    print(f"        FINAL BLEU SCORE: {bleu_score:.2f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
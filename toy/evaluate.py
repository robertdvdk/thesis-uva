# evaluate.py
import torch
import torch.nn as nn
from torchmetrics.text import BLEUScore
import sys
import os
import json
from tokenizers import Tokenizer
from model import Seq2SeqTransformer
from hmodel import HSeq2SeqTransformer
from hypll.manifolds.hyperboloid import Hyperboloid, Curvature
from data import DataPipeline


def translate(model: nn.Module,
              src_tensor: torch.Tensor,
              src_mask: torch.Tensor,
              tokenizer: Tokenizer,
              device: torch.device,
              max_len: int = 50) -> str:
    """
    Translates a source tensor into the target language using greedy decoding.
    """
    model.eval()

    sos_token_id = tokenizer.token_to_id("[SOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")

    with torch.no_grad():
        memory = model.encode(src_tensor, src_key_padding_mask=src_mask)

    ys = torch.ones(1, 1).fill_(sos_token_id).type(torch.long).to(device)
    for i in range(max_len):
        with torch.no_grad():
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device)
            out = model.decode(ys, memory, tgt_mask)
            # Assumes model output has a .tensor attribute, consistent with HSeq2SeqTransformer
            prob = model.generator(out)

            _, next_word = torch.max(prob[:, -1], dim=1)
            next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=1)
        if next_word == eos_token_id:
            break

    translated_tokens = ys[0, 1:].tolist()
    # Remove EOS token from the final output
    if eos_token_id in translated_tokens:
        eos_index = translated_tokens.index(eos_token_id)
        translated_tokens = translated_tokens[:eos_index]

    return tokenizer.decode(translated_tokens)


def calculate_bleu(model: nn.Module,
                   val_loader: torch.utils.data.DataLoader,
                   tokenizer: Tokenizer,
                   device: torch.device) -> float:
    """
    Calculates the BLEU score for the model on the entire validation dataset.
    """
    model.eval()
    candidates, references = [], []
    print("\n--- Calculating BLEU score on the validation set ---")

    with torch.no_grad():
        for batch in val_loader:
            src_tensor_batch = batch['input_ids'].to(device)
            src_mask_batch = batch['attention_mask'].to(device)
            tgt_labels_batch = batch['labels'].tolist()

            # Iterate over each sentence in the batch
            for i in range(src_tensor_batch.size(0)):
                src_tensor = src_tensor_batch[i:i+1]  # Keep batch dim [1, seq_len]
                src_mask = src_mask_batch[i:i+1]    # Keep batch dim [1, seq_len]

                # Prepare the reference sentence
                ref_ids_unpadded = [tid for tid in tgt_labels_batch[i] if tid != -100]
                ref_sentence = tokenizer.decode(ref_ids_unpadded, skip_special_tokens=True)

                # Generate the candidate sentence
                candidate_sentence = translate(model, src_tensor, src_mask, tokenizer, device)
                print(candidate_sentence)
                candidates.append(candidate_sentence)
                references.append([ref_sentence])

    bleu = BLEUScore(n_gram=4)
    score = bleu(candidates, references).item() * 100
    return score


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <path_to_model.pth>")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    if CONFIG["manifold"] == "hyperboloid":
        print("--- Using Hyperboloid manifold ---")
        manifold = Hyperboloid(Curvature(value=CONFIG.get("curvature", 1.0)))

    pipeline = DataPipeline(config=CONFIG)
    _, val_loader, tokenizer = pipeline.get_dataloaders()
    vocab_size = tokenizer.get_vocab_size()

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

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    bleu = calculate_bleu(model, val_loader, tokenizer, device)

    print("-" * 50)
    print(f"              FINAL BLEU SCORE: {bleu:.2f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
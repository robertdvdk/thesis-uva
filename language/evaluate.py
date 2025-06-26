"""
Evaluation script for the Transformer model.

This script loads a trained model checkpoint and evaluates its performance on the
validation set using the BLEU score metric. It uses the torchmetrics library.

Usage:
    python evaluate.py <path_to_model.pth>
"""
import torch
import torch.nn as nn
from torchmetrics.text import BLEUScore
import sys
import os

# --- Import from our project files ---
from model import Seq2SeqTransformer
from data import DataPipeline
from config import CONFIG

def translate(model: nn.Module,
              src_sentence: str,
              tokenizers: dict,
              device: torch.device,
              max_len: int = 50) -> str:
    """
    Translates a source sentence into the target language using greedy decoding.

    Args:
        model (nn.Module): The trained Seq2SeqTransformer model.
        src_sentence (str): The source sentence to translate.
        tokenizers (dict): Dictionary containing 'en' and 'de' tokenizers.
        device (torch.device): The device to run the model on.
        max_len (int): The maximum length for the generated translation.

    Returns:
        str: The translated sentence.
    """
    model.eval()

    # --- 1. Tokenize the source sentence and add special tokens ---
    src_tokenizer = tokenizers[CONFIG['langs'][1]] # e.g., 'en'
    tgt_tokenizer = tokenizers[CONFIG['langs'][0]] # e.g., 'de'

    src_tokens = src_tokenizer.encode(src_sentence).ids
    # Prepend SOS token (id=2) and append EOS token (id=3)
    src_tensor = torch.LongTensor([2] + src_tokens + [3]).unsqueeze(0).to(device)

    # --- 2. Create a source mask ---
    # This mask has 1 for all tokens since there is no padding in a single sentence.
    src_mask = torch.ones(src_tensor.shape, device=device).bool()

    # --- 3. Process source sentence through the encoder ---
    with torch.no_grad():
        memory = model.encode(src_tensor, src_mask)

    # --- 4. Generate the translation using the decoder (greedy search) ---
    # Start with the SOS token
    ys = torch.ones(1, 1).fill_(2).type(torch.long).to(device)

    for i in range(max_len - 1):
        with torch.no_grad():
            # Create a target mask to prevent attending to future tokens
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device)

            # The target padding mask is not needed here as we build the sequence one token at a time
            # The memory_key_padding_mask comes from the original source mask.
            out = model.decode(ys, memory, tgt_mask, memory_key_padding_mask=src_mask)

            # Project to vocabulary and get the token with the highest probability
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

        # Append the predicted token to the sequence
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=1)

        # If the model predicts the EOS token, stop generating
        if next_word == 3:
            break

    # --- 5. Decode the generated token IDs back to a string ---
    # We skip the initial SOS token (index 0)
    translated_tokens = ys[0, 1:].tolist()

    # Remove EOS token if it exists
    if 3 in translated_tokens:
        eos_index = translated_tokens.index(3)
        translated_tokens = translated_tokens[:eos_index]

    return tgt_tokenizer.decode(translated_tokens)


def calculate_bleu(model: nn.Module,
                   val_loader: torch.utils.data.DataLoader,
                   tokenizers: dict,
                   device: torch.device) -> float:
    """
    Calculates the BLEU score for the model on the validation dataset.

    Args:
        model (nn.Module): The trained model.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        tokenizers (dict): The tokenizers for source and target languages.
        device (torch.device): The device to run evaluation on.

    Returns:
        float: The corpus-level BLEU score.
    """
    model.eval()
    candidates = []
    references = []

    src_lang, tgt_lang = CONFIG['langs'][1], CONFIG['langs'][0] # en, de

    print("\n--- Evaluating examples from validation set ---")

    with torch.no_grad():
        # Iterate over the validation set to generate translations
        for i, batch in enumerate(val_loader):
            src_id_lists = batch['input_ids'].tolist()
            tgt_id_lists = batch['labels'].tolist()

            # Decode source sentences
            src_sentences = tokenizers[src_lang].decode_batch(src_id_lists)

            for j in range(len(src_sentences)):
                src_sentence = src_sentences[j]

                # For the target sentence, filter out the padding token's ID (-100) before decoding
                # This prevents the OverflowError and gets the clean reference text directly.
                filtered_tgt_ids = [token_id for token_id in tgt_id_lists[j] if token_id != -100]
                ref_sentence = tokenizers[tgt_lang].decode(filtered_tgt_ids)

                # Generate the model's translation
                candidate_sentence = translate(model, src_sentence, tokenizers, device)

                # Store the full sentences for torchmetrics
                candidates.append(candidate_sentence)
                references.append([ref_sentence]) # Must be a list of lists

                # Print a few examples
                if i < 2 and j < 3: # Print 3 examples from the first 2 batches
                    print("-" * 50)
                    print(f"Source:      {src_sentence}")
                    print(f"Reference:   {ref_sentence}")
                    print(f"Translation: {candidate_sentence}")

    # --- Calculate corpus-level BLEU score using torchmetrics ---
    # n_gram=4 is standard for BLEU
    bleu = BLEUScore(n_gram=4)
    # The score is a tensor, so we extract it with .item() and multiply by 100 for readability
    score = bleu(candidates, references).item() * 100
    return score


def main():
    """
    Main function to load the model and run evaluation.
    """
    # --- 1. Check for model path argument ---
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <path_to_model.pth>")
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # --- 2. Setup device and data pipeline ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    print("--- Initializing Data Pipeline (for tokenizers and validation data) ---")
    # We don't need the training loader here, so we can ignore it
    pipeline = DataPipeline(config=CONFIG)
    _, val_loader, tokenizers = pipeline.get_dataloaders()

    src_vocab_size = tokenizers[CONFIG['langs'][1]].get_vocab_size() # e.g., 'en'
    tgt_vocab_size = tokenizers[CONFIG['langs'][0]].get_vocab_size() # e.g., 'de'

    # --- 3. Load the trained model ---
    print(f"--- Loading model from {model_path} ---")
    model = Seq2SeqTransformer(
        num_encoder_layers=CONFIG["num_encoder_layers"],
        num_decoder_layers=CONFIG["num_decoder_layers"],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=CONFIG["dim_feedforward"],
        dropout=CONFIG["dropout"]
    ).to(device)

    # Load the state dictionary from the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # --- 4. Run evaluation ---
    bleu = calculate_bleu(model, val_loader, tokenizers, device)

    print("-" * 50)
    print(f"              FINAL BLEU SCORE: {bleu:.2f}")
    print("-" * 50)


if __name__ == "__main__":
    main()

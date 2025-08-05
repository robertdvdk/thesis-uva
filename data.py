"""
Contains the DataPipeline class responsible for all data loading,
processing, and preparation for the IWSLT14 dataset.

This version has been adapted to use a single SHARED vocabulary for
tokenization, which is the best practice for English-German translation.
"""

import os
import re
import random
from typing import Tuple, Dict, Any, Iterator, List

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from torch.utils.data import DataLoader, Sampler
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast


class TokenBatchSampler(Sampler[List[int]]):
    """
    A custom sampler that creates batches of indices where each batch
    aims to have a total token count close to a target number.
    """

    def __init__(self, data_source: Dataset, tokens_per_batch: int, shuffle: bool = True):
        super().__init__(data_source)
        self.data_source = data_source
        self.tokens_per_batch = tokens_per_batch
        self.shuffle = shuffle
        self.lengths = [len(x) for x in self.data_source['input_ids']]

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.data_source)))
        # Sort by length for efficiency, forming batches of similar-length sequences
        indices.sort(key=lambda i: self.lengths[i])

        if self.shuffle:
            # Shuffle within pools of similar-sized sequences to maintain some randomness
            pool_size = 1024
            for i in range(0, len(indices), pool_size):
                chunk = indices[i:i + pool_size]
                random.shuffle(chunk)
                indices[i:i + pool_size] = chunk

        batch = []
        max_len_in_batch = 0
        for idx in indices:
            batch.append(idx)
            max_len_in_batch = max(max_len_in_batch, self.lengths[idx])
            tokens_in_batch = len(batch) * max_len_in_batch
            if tokens_in_batch > self.tokens_per_batch:
                yield batch[:-1]
                batch = [idx]
                max_len_in_batch = self.lengths[idx]
        if batch:
            yield batch

    def __len__(self) -> int:
        # An approximation of the number of batches
        return (sum(self.lengths) + self.tokens_per_batch - 1) // self.tokens_per_batch


class DataPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # --- MODIFIED: We now use a single tokenizer object ---
        self.tokenizer: Tokenizer = None
        self.src_lang = 'en'
        self.tgt_lang = 'de'
        self.base_dir = self.config['data_dir_raw']
        self.tmp_dir = os.path.join(self.base_dir, "tmp")
        self.prep_dir = os.path.join(self.base_dir, "prep")
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.prep_dir, exist_ok=True)

    def _clean_and_prepare_raw_files(self):
        """
        Loads the raw dataset, cleans it, filters it, and saves the final
        raw text files for train, valid, and test splits into `self.tmp_dir`.
        This part of the pipeline remains largely the same.
        """
        src, tgt = self.src_lang, self.tgt_lang
        # Check for a final file to see if this whole stage can be skipped
        final_train_file_check = os.path.join(self.tmp_dir, f"train.{src}")
        if os.path.exists(final_train_file_check):
            print("--- Cleaned raw files found in tmp/. Skipping raw data preparation. ---")
            return

        print("--- Starting full raw data preparation pipeline... ---")
        raw_dataset = load_dataset(self.config["dataset_id"], trust_remote_code=True)
        # 1. Extract and clean text from the 'train' split
        for lang in [src, tgt]:
            output_path = os.path.join(self.tmp_dir, f"train.tags.{src}-{tgt}.tok.{lang}")
            with open(output_path, "w", encoding="utf-8") as f:
                for example in raw_dataset['train']:
                    text = example['translation'][lang]
                    text = re.sub(r'<url>.*?</url>', '', text)
                    text = re.sub(r'<talkid>.*?</talkid>', '', text)
                    f.write(text.strip() + "\n")
        # 2. Filter corpus by length and ratio
        self._filter_corpus(
            input_prefix=os.path.join(self.tmp_dir, f"train.tags.{src}-{tgt}.tok"),
            output_prefix=os.path.join(self.tmp_dir, f"train.tags.{src}-{tgt}.clean"),
            src=src, tgt=tgt, min_len=1, max_len=175, ratio=1.5
        )
        # 3. Lowercase and create train/validation splits
        for lang in [src, tgt]:
            clean_path = os.path.join(self.tmp_dir, f"train.tags.{src}-{tgt}.clean.{lang}")
            final_path = os.path.join(self.tmp_dir, f"train.tags.{src}-{tgt}.{lang}")
            with open(clean_path, "r", encoding="utf-8") as fin, open(final_path, "w", encoding="utf-8") as fout:
                for line in fin: fout.write(line.lower())

            with open(final_path, "r", encoding="utf-8") as fin, \
                    open(os.path.join(self.tmp_dir, f"train.{lang}"), "w", encoding="utf-8") as f_train, \
                    open(os.path.join(self.tmp_dir, f"valid.{lang}"), "w", encoding="utf-8") as f_valid:
                for i, line in enumerate(fin):
                    # Simple split: every 23rd line goes to validation
                    if i % 23 == 0:
                        f_valid.write(line)
                    else:
                        f_train.write(line)
        # 4. Prepare test set from raw validation and test splits
        for lang in [src, tgt]:
            with open(os.path.join(self.tmp_dir, f"test.{lang}"), "w", encoding="utf-8") as f_test:
                for example in raw_dataset['validation']: f_test.write(
                    example['translation'][lang].strip().lower() + "\n")
                for example in raw_dataset['test']: f_test.write(example['translation'][lang].strip().lower() + "\n")
        print("--- Raw data preparation complete. ---")

    def _filter_corpus(self, input_prefix: str, output_prefix: str, src: str, tgt: str, min_len: int, max_len: int,
                       ratio: float):
        src_in, tgt_in = f"{input_prefix}.{src}", f"{input_prefix}.{tgt}"
        src_out, tgt_out = f"{output_prefix}.{src}", f"{output_prefix}.{tgt}"
        with open(src_in, "r", encoding="utf-8") as f_src_in, open(tgt_in, "r", encoding="utf-8") as f_tgt_in, \
                open(src_out, "w", encoding="utf-8") as f_src_out, open(tgt_out, "w", encoding="utf-8") as f_tgt_out:
            for src_line, tgt_line in zip(f_src_in, f_tgt_in):
                src_sent, tgt_sent = src_line.strip(), tgt_line.strip()
                if not src_sent or not tgt_sent: continue
                src_len, tgt_len = len(src_sent.split()), len(tgt_sent.split())
                if not (min_len <= src_len <= max_len and min_len <= tgt_len <= max_len): continue
                if src_len > 0 and tgt_len > 0 and max(src_len, tgt_len) / min(src_len, tgt_len) <= ratio:
                    f_src_out.write(src_sent + "\n")
                    f_tgt_out.write(tgt_sent + "\n")

    # --- NEW METHOD: Replaces the two old BPE/vocab methods ---
    def _build_shared_tokenizer(self):
        """
        Trains and saves a single shared BPE tokenizer on the combined
        source and target training data.
        """
        tokenizer_path = os.path.join(self.prep_dir, "shared_bpe_tokenizer.json")
        if os.path.exists(tokenizer_path):
            print("--- Loading existing shared tokenizer. ---")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            return

        print("--- Training a new shared BPE tokenizer... ---")
        # Train on the concatenated train files for both languages
        train_files = [os.path.join(self.tmp_dir, f"train.{lang}") for lang in [self.src_lang, self.tgt_lang]]

        # Check if training files exist before proceeding
        for f in train_files:
            if not os.path.exists(f):
                raise FileNotFoundError(
                    f"Training file not found: {f}. Ensure _clean_and_prepare_raw_files() has been run.")

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=self.config["vocab_size"],
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        tokenizer.train(train_files, trainer=trainer)
        tokenizer.save(tokenizer_path)
        self.tokenizer = tokenizer
        print("--- Shared tokenizer training complete. ---")

    # --- MODIFIED: Now loads raw text and tokenizes on-the-fly ---
    def _load_final_dataset(self) -> DatasetDict:
        """
        Loads the prepared raw text files from `tmp_dir`, tokenizes them
        on the fly, and returns a tokenized DatasetDict.
        """
        src, tgt = self.src_lang, self.tgt_lang

        # Load the raw text from the cleaned files in tmp_dir
        datasets = {}
        for split in ["train", "validation", "test"]:
            split_name = "valid" if split == "validation" else split
            ds_src = load_dataset("text", data_files=os.path.join(self.tmp_dir, f"{split_name}.{src}"))['train']
            ds_tgt = load_dataset("text", data_files=os.path.join(self.tmp_dir, f"{split_name}.{tgt}"))['train']
            # Combine the two text datasets side-by-side
            datasets[split] = concatenate_datasets(
                [ds_src.rename_column("text", src), ds_tgt.rename_column("text", tgt)], axis=1)
        raw_dataset = DatasetDict(datasets)

        # Set the post-processor for the single shared tokenizer
        sos_token_id = self.tokenizer.token_to_id("[SOS]")
        eos_token_id = self.tokenizer.token_to_id("[EOS]")
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[("[SOS]", sos_token_id), ("[EOS]", eos_token_id)],
        )

        def tokenize_function(examples):
            # Use the SAME tokenizer for both source and target
            source_tokenized = self.tokenizer.encode_batch(examples[src])
            target_tokenized = self.tokenizer.encode_batch(examples[tgt])
            return {
                "input_ids": [enc.ids for enc in source_tokenized],
                "attention_mask": [enc.attention_mask for enc in source_tokenized],
                "labels": [enc.ids for enc in target_tokenized]
            }

        print("--- Tokenizing datasets on the fly... ---")
        tokenized_datasets = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[src, tgt]
        )
        return tokenized_datasets

    # --- MODIFIED: Updated pipeline flow and return value ---
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, Tokenizer]:
        """
        The main method to orchestrate the entire data pipeline.
        Returns DataLoaders for train/validation and the shared tokenizer.
        """
        # 1. Prepare raw text files if they don't exist
        self._clean_and_prepare_raw_files()

        # 2. Build or load the single shared tokenizer
        self._build_shared_tokenizer()

        # 3. Load and tokenize the datasets on the fly
        tokenized_datasets = self._load_final_dataset()

        # Wrap the single tokenizer for the transformers library
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
            bos_token="[SOS]",
            eos_token="[EOS]"
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=fast_tokenizer,
            padding="longest",
            return_tensors="pt"
        )

        # Use the custom token-based sampler
        train_sampler = TokenBatchSampler(tokenized_datasets["train"], tokens_per_batch=self.config["tokens_per_batch"],
                                          shuffle=True)
        val_sampler = TokenBatchSampler(tokenized_datasets["validation"],
                                        tokens_per_batch=self.config["tokens_per_batch"], shuffle=False)

        train_loader = DataLoader(tokenized_datasets["train"], batch_sampler=train_sampler, collate_fn=data_collator)
        val_loader = DataLoader(tokenized_datasets["validation"], batch_sampler=val_sampler, collate_fn=data_collator)

        # Return the single shared tokenizer instead of a dictionary
        return train_loader, val_loader, self.tokenizer
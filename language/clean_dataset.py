import os
import re
from datasets import load_dataset, Dataset, concatenate_datasets

# --- 1. Set up parameters (equivalent to your command line arguments) ---
# Language pair extensions
src_lang = "src"
tgt_lang = "tgt"

# Base path for input files
corpus_base_path = f"$tmp/train.tags.{src_lang}.tok"  # Use your actual path

# Input file paths
source_file = f"{corpus_base_path}.{src_lang}"
target_file = f"{corpus_base_path}.{tgt_lang}"

# Base path for output files
output_base_path = f"$tmp/train.tags.{src_lang}.clean"  # Use your actual path
cleaned_source_file = f"{output_base_path}.{src_lang}"
cleaned_target_file = f"{output_base_path}.{tgt_lang}"

# Filtering criteria
min_len = 1
max_len = 175
ratio_threshold = 1.5

# --- 2. Load the parallel text files into a single Dataset object ---

# load_dataset can load text files line-by-line
ds_src = load_dataset("text", data_files={"train": source_file})["train"]
ds_tgt = load_dataset("text", data_files={"train": target_file})["train"]

# Rename the default 'text' column to be specific
ds_src = ds_src.rename_column("text", src_lang)
ds_tgt = ds_tgt.rename_column("text", tgt_lang)

# Combine them into one dataset with two columns: 'src' and 'tgt'
# This assumes the files have the same number of lines
parallel_dataset = concatenate_datasets([ds_src, ds_tgt], axis=1)

print(f"Initial dataset size: {len(parallel_dataset)}")


# --- 3. Define the filtering function (the core logic) ---
def clean_example(example):
    """
    This function replicates the filtering logic of clean-corpus-n.perl
    It returns True to keep the example, False to discard it.
    """
    source_sentence = example[src_lang]
    target_sentence = example[tgt_lang]

    # The Perl script normalizes whitespace first
    source_sentence = re.sub(r'\s+', ' ', source_sentence).strip()
    target_sentence = re.sub(r'\s+', ' ', target_sentence).strip()

    # Get word counts (simple split on space, like the Perl script)
    src_words = source_sentence.split(' ')
    tgt_words = target_sentence.split(' ')
    src_len = len(src_words)
    tgt_len = len(tgt_words)

    # Filter by emptiness (will also be caught by min_len, but good to be explicit)
    if src_len == 0 or tgt_len == 0:
        return False

    # Filter by min/max length
    if not (min_len <= src_len <= max_len):
        return False
    if not (min_len <= tgt_len <= max_len):
        return False

    # Filter by ratio
    # Ensure no division by zero, although the length check above handles it
    ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
    if ratio > ratio_threshold:
        return False

    # (Optional) The Perl script also checks for max word length (default 1000)
    # This is less common but can be added if needed
    # max_word_len_in_src = max(len(w) for w in src_words) if src_len > 0 else 0
    # max_word_len_in_tgt = max(len(w) for w in tgt_words) if tgt_len > 0 else 0
    # if max_word_len_in_src > 1000 or max_word_len_in_tgt > 1000:
    #    return False

    return True


# --- 4. Apply the filter ---
# The .filter() method will iterate through the dataset and keep only
# the examples for which the function returns True.
cleaned_dataset = parallel_dataset.filter(clean_example)

print(f"Cleaned dataset size: {len(cleaned_dataset)}")


# --- 5. Save the cleaned data back to text files ---
def save_column_to_file(dataset, column_name, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(example[column_name] + "\n")


# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

save_column_to_file(cleaned_dataset, src_lang, cleaned_source_file)
save_column_to_file(cleaned_dataset, tgt_lang, cleaned_target_file)

print(f"Cleaned files saved to:\n- {cleaned_source_file}\n- {cleaned_target_file}")
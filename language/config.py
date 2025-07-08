# config.py

"""
Central configuration file for the Transformer project.
"""

CONFIG = {
    # --- Run Identification & Checkpointing ---
    "run_name": "hyp_test",
    "load_model_path": None,

    # --- Data & Tokenizer Parameters ---
    "dataset_id": "bbaaaa/iwslt14-de-en",
    "data_dir_raw": "iwslt14_data_raw",
    "langs": ['de', 'en'],
    "vocab_size": 10000,

    # --- Model Architecture Parameters (Aligned with Paper) ---
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "dim_feedforward": 256,
    "dropout": 0.1,

    # --- Geometry Configuration ---
    "manifold": "Euclidean",  # Options: "Euclidean" or "hyperboloid"
    "curvature": 1.0,

    # --- Training & Optimization Parameters (Step-Based) ---
    # --- MODIFIED: from epochs to steps ---
    "max_steps": 40000,           # Total number of training steps
    "eval_interval": 1000,        # Evaluate every N steps
    "label_smoothing": 0.1,
    "tokens_per_batch": 10240,
    "max_grad_norm": 0.5,

    # --- Learning Rate Scheduler ("Noam" Style) ---
    "lr_multiplier": 5.0,
    "warmup_steps": 6000,
}
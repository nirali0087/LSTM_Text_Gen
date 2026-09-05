import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("KERAS_BACKEND", "torch")

# Flexible import
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    import keras

# Reuse preprocessing from Task 1 and vocabulary loading from Task 2
from src.preprocess import clean_text, tokenize_text
from src.prepare_data import load_vocabulary


def load_generation_resources(
    model_path: Union[str, Path] = "models/best_model.keras",
    vocab_dir: Union[str, Path] = "models",
) -> Tuple[keras.Model, Dict[str, int], Dict[int, str]]:
    """
    Loads the trained model and vocabulary mappings from disk.
    Model is loaded once and reused across all generation queries.

    Args:
        model_path (Union[str, Path]): Path to the best saved model file.
        vocab_dir (Union[str, Path]): Directory containing token_to_id.json and id_to_token.json.

    Returns:
        Tuple[keras.Model, Dict[str, int], Dict[int, str]]:
            - loaded model
            - token_to_id mapping
            - id_to_token mapping
    """
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Trained model file not found at: {model_file}. "
            "Please train the model first by running Task 4."
        )

    model = keras.models.load_model(str(model_file))
    token_to_id, id_to_token = load_vocabulary(vocab_dir=vocab_dir)

    return model, token_to_id, id_to_token


def sample_next_token(
    probabilities: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> int:
    """
    Samples the next token index from a probability distribution using temperature
    scaling and optional top-k filtering.

    Temperature behavior:
    - temperature < 1.0: sharper distribution, more deterministic/conservative predictions.
    - temperature == 1.0: original model probability distribution.
    - temperature > 1.0: softer distribution, more creative/diverse tokens.

    Args:
        probabilities (np.ndarray): 1D array of softmax probabilities over vocabulary.
        temperature (float): Positive temperature value. Default is 1.0.
        top_k (Optional[int]): If provided, limits sampling to the top K most probable tokens.

    Returns:
        int: Selected token ID index.
    """
    # Deterministic greedy argmax for near-zero temperature
    if temperature <= 1e-5:
        return int(np.argmax(probabilities))

    # Numerically stable log-space transformation
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    logits = np.log(probs) / temperature

    # Subtract max for numerical stability against exponent overflow
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)

    # Optional top-k filtering
    if top_k is not None and 0 < top_k < len(exp_logits):
        top_indices = np.argsort(exp_logits)[-top_k:]
        filtered = np.zeros_like(exp_logits)
        filtered[top_indices] = exp_logits[top_indices]
        exp_logits = filtered

    # Normalize to obtain probability distribution
    prob_sum = np.sum(exp_logits)
    if prob_sum <= 0 or np.isnan(prob_sum) or np.isinf(prob_sum):
        return int(np.argmax(probabilities))

    p = exp_logits / prob_sum

    # Sample from multinomial distribution
    try:
        return int(np.random.choice(len(p), p=p))
    except Exception:
        return int(np.argmax(probabilities))


def generate_text(
    seed_text: str,
    num_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    model: Optional[keras.Model] = None,
    token_to_id: Optional[Dict[str, int]] = None,
    id_to_token: Optional[Dict[int, str]] = None,
    model_path: Union[str, Path] = "models/best_model.keras",
    vocab_dir: Union[str, Path] = "models",
    sequence_length: int = 20,
) -> str:
    """
    Generates text continuation given a seed text prompt.

    Pipeline:
    1. Preprocesses the seed prompt using Task 1 logic (lowercase, remove punctuation, normalize whitespace).
    2. Converts tokens to integer IDs using the Task 2 vocabulary.
    3. Handles unknown words gracefully without crashing or fabricating an <UNK> token.
    4. Recursively predicts the next token, appends it, and slides the 20-token context window forward.
    5. Decodes the generated tokens back to words and returns the complete text.

    Args:
        seed_text (str): Input prompt string.
        num_tokens (int): Number of new tokens to generate. Default is 50.
        temperature (float): Sampling temperature. Default is 1.0.
        top_k (Optional[int]): Optional top-k probability filter.
        model (Optional[keras.Model]): Preloaded model instance. If None, loads from model_path.
        token_to_id (Optional[Dict[str, int]]): Token-to-ID mapping. If None, loads from vocab_dir.
        id_to_token (Optional[Dict[int, str]]): ID-to-token mapping. If None, loads from vocab_dir.
        model_path (Union[str, Path]): Path to saved model file.
        vocab_dir (Union[str, Path]): Directory containing vocabulary files.
        sequence_length (int): Context window length. Default is 20.

    Returns:
        str: Generated text continuation.
    """
    # Load resources once if not passed
    if model is None or token_to_id is None or id_to_token is None:
        model, token_to_id, id_to_token = load_generation_resources(
            model_path=model_path, vocab_dir=vocab_dir
        )

    # 1. Clean and tokenize seed text with Task 1 pipeline
    cleaned_seed = clean_text(seed_text)
    seed_tokens = tokenize_text(cleaned_seed)

    if not seed_tokens:
        print("Warning: Empty seed text provided. Using default start token 'the'.")
        seed_tokens = ["the"]

    # 2. Convert to integer IDs & handle unknown tokens gracefully
    known_ids: List[int] = []
    unknown_tokens: List[str] = []

    for token in seed_tokens:
        if token in token_to_id:
            known_ids.append(token_to_id[token])
        else:
            unknown_tokens.append(token)

    if unknown_tokens:
        print(f"[Notice] Ignored {len(unknown_tokens)} out-of-vocabulary seed token(s): {unknown_tokens}")

    if not known_ids:
        # Fallback to most frequent token 'the' (id 0) if all seed words were unknown
        print("[Notice] All seed tokens were out-of-vocabulary. Falling back to 'the'.")
        known_ids = [0]

    current_ids = list(known_ids)
    generated_tokens: List[str] = []

    # 3. Autoregressive generation loop
    for _ in range(num_tokens):
        # Format the 20-token receptive field
        if len(current_ids) < sequence_length:
            # If seed is shorter than sequence_length, repeat cyclically so the prompt
            # fills the 20-token window ending with the exact seed tokens
            multiplier = (sequence_length // len(current_ids)) + 1
            input_window = (current_ids * multiplier)[-sequence_length:]
        else:
            input_window = current_ids[-sequence_length:]

        input_batch = np.array([input_window], dtype=np.int32)

        # Predict next-token probability distribution
        probabilities = model.predict(input_batch, verbose=0)[0]

        # Sample next token
        next_id = sample_next_token(probabilities, temperature=temperature, top_k=top_k)

        # Append and record
        current_ids.append(next_id)
        next_token_str = id_to_token.get(next_id, "")
        generated_tokens.append(next_token_str)

    # Decode and format output
    generated_text = " ".join(generated_tokens)
    return generated_text


def main() -> None:
    """
    Validates text generation on the mandatory Shakespeare seeds.
    """
    model_path = Path("models/best_model.keras")
    vocab_dir = Path("models")

    print("=" * 65)
    print("TASK 5: TEXT GENERATION VALIDATION")
    print("=" * 65)

    if not model_path.is_file():
        print(f"\nModel file not found at '{model_path}'.")
        print("Training model now for Task 5 generation...\n")
        from src.model import train_model
        train_model()

    print(f"\nLoading trained model from '{model_path}'...")
    model, token_to_id, id_to_token = load_generation_resources(
        model_path=model_path, vocab_dir=vocab_dir
    )
    print("Model and vocabulary loaded successfully.")

    # Test seeds specified in the prompt
    test_seeds = [
        "to be or not to be",
        "shall i compare thee",
        "all the world's a stage",
    ]

    temperatures = [0.7, 1.0]

    for seed in test_seeds:
        print("\n" + "-" * 65)
        print(f"Seed: \"{seed}\"")
        for temp in temperatures:
            generated = generate_text(
                seed_text=seed,
                num_tokens=40,
                temperature=temp,
                model=model,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
            )
            print(f"\nGenerated text (temperature={temp}):")
            print(f"{seed} {generated}")

    print("\n" + "=" * 65)
    print("Text generation completed successfully.")


if __name__ == "__main__":
    main()

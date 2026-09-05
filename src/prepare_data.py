from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.preprocess import clean_text, load_text, tokenize_text
except ImportError:
    from preprocess import clean_text, load_text, tokenize_text


def build_vocabulary(
    tokens: List[str],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds deterministic bidirectional vocabulary mappings from a list of tokens.
    Tokens are sorted by frequency (descending) with alphabetical tie-breaking
    to ensure full reproducibility.

    Args:
        tokens (List[str]): Cleaned word tokens from the corpus.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]:
            - token_to_id: Mapping from token string to integer ID.
            - id_to_token: Mapping from integer ID to token string.
    """
    counts = Counter(tokens)
    # Sort by frequency descending; use token string ascending as tie-breaker
    sorted_tokens = sorted(counts.keys(), key=lambda word: (-counts[word], word))

    token_to_id = {token: idx for idx, token in enumerate(sorted_tokens)}
    id_to_token = {idx: token for idx, token in enumerate(sorted_tokens)}

    return token_to_id, id_to_token


def encode_tokens(tokens: List[str], token_to_id: Dict[str, int]) -> List[int]:
    """
    Converts a list of word tokens into their corresponding integer IDs.

    Args:
        tokens (List[str]): List of string tokens.
        token_to_id (Dict[str, int]): Token-to-ID mapping dictionary.

    Returns:
        List[int]: List of integer token IDs.
    """
    return [token_to_id[token] for token in tokens]


def decode_ids(
    token_ids: Union[List[int], np.ndarray],
    id_to_token: Dict[int, str],
) -> List[str]:
    """
    Converts integer token IDs back into their original string tokens.

    Args:
        token_ids (Union[List[int], np.ndarray]): Sequence of integer token IDs.
        id_to_token (Dict[int, str]): ID-to-token mapping dictionary.

    Returns:
        List[str]: List of decoded string tokens.
    """
    return [id_to_token[int(idx)] for idx in token_ids]


def create_sequences(
    token_ids: Union[List[int], np.ndarray],
    sequence_length: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates input-target sequence pairs using a sliding window.
    For each window of length `sequence_length`, the target is the token
    immediately following the window.

    Example with sequence_length = 3:
        Input:  [w1, w2, w3] -> Target: w4
        Input:  [w2, w3, w4] -> Target: w5

    Args:
        token_ids (Union[List[int], np.ndarray]): Sequence of integer token IDs.
        sequence_length (int): Number of tokens in each input sequence. Default is 20.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: Array of shape (num_sequences, sequence_length), dtype int32.
            - y: Array of shape (num_sequences,), dtype int32.

    Raises:
        ValueError: If sequence_length is not positive or exceeds token length.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    arr = np.asarray(token_ids, dtype=np.int32)
    num_sequences = len(arr) - sequence_length

    if num_sequences <= 0:
        raise ValueError(
            f"Token length ({len(arr)}) must be greater than sequence_length ({sequence_length})"
        )

    # Efficient vectorised sliding window using NumPy stride tricks
    windows = sliding_window_view(arr, window_shape=sequence_length + 1)
    X = np.array(windows[:, :-1], copy=True, dtype=np.int32)
    y = np.array(windows[:, -1], copy=True, dtype=np.int32)

    return X, y


def save_vocabulary(
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    output_dir: Union[str, Path] = "models",
) -> Tuple[Path, Path]:
    """
    Saves vocabulary mappings to JSON files in the specified directory.
    Handles integer dictionary keys properly during JSON serialization.

    Files created:
        - {output_dir}/token_to_id.json
        - {output_dir}/id_to_token.json

    Args:
        token_to_id (Dict[str, int]): Token-to-ID mapping.
        id_to_token (Dict[int, str]): ID-to-token mapping.
        output_dir (Union[str, Path]): Target directory for vocabulary files.

    Returns:
        Tuple[Path, Path]: Paths to the saved files (token_to_id_path, id_to_token_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token_to_id_path = out_dir / "token_to_id.json"
    id_to_token_path = out_dir / "id_to_token.json"

    with open(token_to_id_path, "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, indent=2, ensure_ascii=False)

    # JSON requires keys to be strings; serialize integer keys as strings
    serialized_id_to_token = {str(k): v for k, v in id_to_token.items()}
    with open(id_to_token_path, "w", encoding="utf-8") as f:
        json.dump(serialized_id_to_token, f, indent=2, ensure_ascii=False)

    return token_to_id_path, id_to_token_path


def load_vocabulary(
    vocab_dir: Union[str, Path] = "models",
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Loads vocabulary mappings from JSON files.
    Ensures integer dictionary keys in id_to_token are correctly converted back
    from JSON strings into Python integers.

    Args:
        vocab_dir (Union[str, Path]): Directory containing token_to_id.json and id_to_token.json.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]:
            - token_to_id: Dict mapping token string to integer ID.
            - id_to_token: Dict mapping integer ID to token string.
    """
    v_dir = Path(vocab_dir)
    token_to_id_path = v_dir / "token_to_id.json"
    id_to_token_path = v_dir / "id_to_token.json"

    if not token_to_id_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {token_to_id_path}")
    if not id_to_token_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {id_to_token_path}")

    with open(token_to_id_path, "r", encoding="utf-8") as f:
        token_to_id: Dict[str, int] = json.load(f)

    with open(id_to_token_path, "r", encoding="utf-8") as f:
        raw_id_to_token: Dict[str, str] = json.load(f)

    # Convert stringified JSON keys back into integers
    id_to_token: Dict[int, str] = {int(k): v for k, v in raw_id_to_token.items()}

    return token_to_id, id_to_token


def prepare_dataset(
    tokens: Optional[List[str]] = None,
    sequence_length: int = 20,
    train_ratio: float = 0.8,
    data_path: Union[str, Path] = "data/shakespeare.txt",
    save_vocab: bool = True,
    vocab_dir: Union[str, Path] = "models",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    End-to-end data preparation pipeline:
    1. Loads and preprocesses tokens if not provided.
    2. Builds vocabulary (token_to_id and id_to_token).
    3. Encodes tokens into integer IDs.
    4. The token corpus is split chronologically before creating training and
       validation sequences, preventing overlapping sliding-window sequences
       from crossing the train-validation boundary.
    5. Creates (X, y) sliding-window sequence pairs for train and validation.
    6. Saves vocabulary files to disk if save_vocab is True.

    Args:
        tokens (Optional[List[str]]): Preprocessed tokens. If None, loaded from data_path.
        sequence_length (int): Length of each input sequence window. Default is 20.
        train_ratio (float): Fraction of tokens to allocate to training. Default is 0.8.
        data_path (Union[str, Path]): Path to raw text dataset. Default is 'data/shakespeare.txt'.
        save_vocab (bool): Whether to save vocabulary mappings to JSON. Default is True.
        vocab_dir (Union[str, Path]): Directory to save vocabulary files. Default is 'models'.

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, token_to_id, id_to_token)
    """
    # 1. Obtain tokens using Task 1 preprocessing
    if tokens is None:
        raw_text = load_text(data_path)
        cleaned_text = clean_text(raw_text)
        tokens = tokenize_text(cleaned_text)

    # 2. Build vocabulary
    token_to_id, id_to_token = build_vocabulary(tokens)

    # 3. Save vocabulary if requested
    if save_vocab:
        save_vocabulary(token_to_id, id_to_token, output_dir=vocab_dir)

    # 4. Integer encode all tokens
    encoded_ids = np.array(encode_tokens(tokens, token_to_id), dtype=np.int32)

    # 5. Chronological Train/Validation Split:
    # The token corpus is split chronologically before creating training and validation sequences,
    # preventing overlapping sliding-window sequences from crossing the train-validation boundary.
    split_idx = int(len(encoded_ids) * train_ratio)
    train_ids = encoded_ids[:split_idx]
    val_ids = encoded_ids[split_idx:]

    # 6. Generate sliding-window sequences
    X_train, y_train = create_sequences(train_ids, sequence_length=sequence_length)
    X_val, y_val = create_sequences(val_ids, sequence_length=sequence_length)

    return X_train, y_train, X_val, y_val, token_to_id, id_to_token


def main() -> None:
    """
    Executes Task 2 pipeline and prints required dataset statistics
    and a real Shakespeare example.
    """
    data_path = Path("data/shakespeare.txt")
    vocab_dir = Path("models")
    sequence_length = 20
    train_ratio = 0.8

    print("=" * 60)
    print("TASK 2: VOCABULARY & SEQUENCE PREPARATION")
    print("=" * 60)

    # 1. Load and tokenize using Task 1 pipeline
    print(f"\n[1/5] Loading and tokenizing from '{data_path}'...")
    raw_text = load_text(data_path)
    cleaned_text = clean_text(raw_text)
    tokens = tokenize_text(cleaned_text)
    total_tokens = len(tokens)
    print(f"      Total tokens extracted: {total_tokens:,}")

    # 2. Build vocabulary
    print("\n[2/5] Building deterministic vocabulary...")
    token_to_id, id_to_token = build_vocabulary(tokens)
    vocab_size = len(token_to_id)
    print(f"      Vocabulary size: {vocab_size:,} unique tokens")

    # 3. Save vocabulary
    print(f"\n[3/5] Saving vocabulary to '{vocab_dir}/'...")
    t2id_path, id2t_path = save_vocabulary(token_to_id, id_to_token, output_dir=vocab_dir)
    print(f"      Saved: {t2id_path}")
    print(f"      Saved: {id2t_path}")

    # Verify JSON load with integer key preservation
    loaded_t2id, loaded_id2t = load_vocabulary(vocab_dir=vocab_dir)
    assert len(loaded_t2id) == vocab_size, "Mismatch in loaded token_to_id size!"
    assert len(loaded_id2t) == vocab_size, "Mismatch in loaded id_to_token size!"
    assert all(isinstance(k, int) for k in loaded_id2t.keys()), "id_to_token keys must be int!"
    print("      Verification: Vocabulary successfully reloaded with int keys intact.")

    # 4. Integer encode tokens
    print("\n[4/5] Encoding tokens into integer IDs...")
    encoded_ids = np.array(encode_tokens(tokens, token_to_id), dtype=np.int32)
    print(f"      Encoded {len(encoded_ids):,} tokens.")

    # 5. Chronological Train / Validation Split (80% / 20%)
    print(f"\n[5/5] Creating next-token sequences (sequence_length={sequence_length})...")
    split_idx = int(len(encoded_ids) * train_ratio)
    train_ids = encoded_ids[:split_idx]
    val_ids = encoded_ids[split_idx:]

    X_train, y_train = create_sequences(train_ids, sequence_length=sequence_length)
    X_val, y_val = create_sequences(val_ids, sequence_length=sequence_length)

    total_sequences = len(X_train) + len(X_val)
    training_sequences = len(X_train)
    validation_sequences = len(X_val)

    # Print Dataset Statistics
    print("\n" + "=" * 60)
    print("DATASET PREPARATION STATISTICS")
    print("=" * 60)
    print(f"total tokens         : {total_tokens}")
    print(f"vocabulary size      : {vocab_size}")
    print(f"sequence length      : {sequence_length}")
    print(f"total sequences      : {total_sequences}")
    print(f"training sequences   : {training_sequences}")
    print(f"validation sequences : {validation_sequences}")
    print(f"X_train shape        : {X_train.shape}")
    print(f"y_train shape        : {y_train.shape}")
    print(f"X_val shape          : {X_val.shape}")
    print(f"y_val shape          : {y_val.shape}")
    print("=" * 60)

    # Real Shakespeare Example from X_train[0] and y_train[0]
    example_input_ids = X_train[0].tolist()
    example_input_tokens = decode_ids(example_input_ids, id_to_token)
    example_target_id = int(y_train[0])
    example_target_token = id_to_token[example_target_id]

    print("\nREAL SHAKESPEARE EXAMPLE:")
    print("=" * 60)
    print(f"Input token IDs:\n{example_input_ids}")
    print(f"\nInput tokens:\n{example_input_tokens}")
    print(f"\nTarget token ID:\n{example_target_id}")
    print(f"\nTarget token:\n'{example_target_token}'")
    print("=" * 60)


if __name__ == "__main__":
    main()

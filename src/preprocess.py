from pathlib import Path
import re
import sys
from typing import List, Union


def load_text(filepath: Union[str, Path] = "data/shakespeare.txt") -> str:
    """
    Loads the raw text dataset from the specified file path.
    Keeps the original dataset file completely unchanged (read-only mode).

    Args:
        filepath (Union[str, Path]): Path to the text file. Defaults to 'data/shakespeare.txt'.

    Returns:
        str: Raw text content of the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(filepath)

    if not path.is_file():
        project_root = Path(__file__).resolve().parent.parent
        resolved_path = project_root / filepath
        if resolved_path.is_file():
            path = resolved_path
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}' or '{resolved_path}'."
            )

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def clean_text(text: str) -> str:
    """
    Cleans raw text through the following pipeline:
    1. Converts all characters to lowercase.
    2. Removes all punctuation marks (ASCII punctuation, unicode quotes/dashes/symbols,
       and underscores which Gutenberg uses for italic formatting).
    3. Normalizes unnecessary whitespace (converts sequences of spaces, tabs,
       and newlines into a single space and strips boundaries).

    Args:
        text (str): Raw or partially processed text string.

    Returns:
        str: Cleaned and normalized text string.
    """
    text = text.lower()

    text = re.sub(r"[^\w\s]|_", "", text)

    text = " ".join(text.split())

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes cleaned text into a list of individual word tokens.

    Args:
        text (str): Cleaned and normalized text.

    Returns:
        List[str]: List of word tokens.
    """
    return text.split()


def preprocess_text(text: str) -> List[str]:
    """
    End-to-end reusable preprocessing pipeline:
    raw text -> lowercase -> punctuation removal -> whitespace normalization -> word tokens.

    This function is designed to be reused directly during text generation to preprocess
    seed/prompt text using the exact same logic.

    Args:
        text (str): Raw input text string.

    Returns:
        List[str]: List of preprocessed word tokens.
    """
    cleaned = clean_text(text)
    return tokenize_text(cleaned)


def main() -> None:
    """
    Executes the Task 1 preprocessing pipeline on the Shakespeare dataset
    and prints real statistics.
    """
    data_path = Path("data/shakespeare.txt")

    print("=" * 60)
    print("Shakespeare Text Preprocessing")
    print("=" * 60)

    print(f"\nLoading dataset from '{data_path}'...")
    raw_text = load_text(data_path)
    original_char_count = len(raw_text)
    print(f"Loaded {original_char_count:,} characters.")

    print("\nCleaning the text...")
    cleaned_text = clean_text(raw_text)
    cleaned_char_count = len(cleaned_text)
    print(f"After cleaning: {cleaned_char_count:,} characters.")

    print("\nSplitting the text into words...")
    tokens = tokenize_text(cleaned_text)
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    print(f"Total words: {total_tokens:,}")
    print(f"Unique words: {unique_tokens:,}")

    first_50_tokens = tokens[:50]

    print("\n" + "=" * 60)
    print("Preprocessing Results")
    print("=" * 60)
    print(f"Original characters : {original_char_count}")
    print(f"Cleaned characters  : {cleaned_char_count}")
    print(f"Total tokens        : {total_tokens}")
    print(f"Unique tokens       : {unique_tokens}")
    print("=" * 60)

    print("\nFirst 50 tokens:")
    print(first_50_tokens)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

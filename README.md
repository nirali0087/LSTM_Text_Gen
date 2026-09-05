# LSTM Text Generator

## Objective
This project implements a generative text generation system using a Long Short-Term Memory (LSTM) recurrent neural network. The model is designed to learn linguistic patterns, style, and sequential dependencies from training data, allowing it to generate coherent text continuations when provided with a seed text prompt.

## Dataset
The dataset consists of William Shakespeare's public-domain complete works. The text was obtained from [Project Gutenberg](https://www.gutenberg.org/) and contains classic plays, sonnets, and poems stored as raw UTF-8 text in `data/shakespeare.txt`.

## Task 1 - Preprocessing
Before feeding textual data to a sequential neural model, raw text must be converted into a structured, normalized token sequence. The preprocessing pipeline in `src/preprocess.py` transforms the raw text through four stages:

```
Raw text
→ lowercase
→ punctuation removal
→ whitespace normalization
→ word tokenization
```

1. **Raw text loading**: The dataset is loaded in read-only mode, keeping the source data intact.
2. **Lowercase conversion**: All characters are converted to lowercase to standardize vocabulary representation (e.g., treating "Thou" and "thou" identically).
3. **Punctuation removal**: All punctuation marks—including ASCII punctuation, typographic quotes (`‘`, `’`, `“`, `”`), dashes (`—`), and Gutenberg italic formatting underscores (`_`)—are removed to prevent vocabulary fragmentation.
4. **Whitespace normalization**: Multiple spaces, tabs, and newline characters are collapsed into a single space, and boundary whitespace is stripped.
5. **Word tokenization**: The normalized text is split into a sequential list of individual word tokens.

This preprocessing logic is fully encapsulated in modular and reusable functions (`load_text`, `clean_text`, `tokenize_text`, and `preprocess_text`) to ensure identical tokenization is applied to seed prompts during future text generation.

### Running Task 1 Preprocessing

To execute the preprocessing pipeline and inspect dataset statistics:

```bash
python src/preprocess.py
```

### Preprocessing Statistics

Running the pipeline on `data/shakespeare.txt` yields the following verified statistics:

| Metric | Value |
| :--- | :--- |
| **Original character count** | 5,378,701 |
| **Cleaned character count** | 5,040,484 |
| **Total number of tokens** | 966,377 |
| **Number of unique tokens** | 30,536 |

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

## Task 2 - Sequence Preparation
Once raw text has been preprocessed into tokens, it is transformed into numerical training examples suitable for sequential next-token learning in `src/prepare_data.py`:

```
Raw text
→ preprocessing
→ tokens
→ vocabulary
→ integer IDs
→ fixed-length sequences
→ next-token targets
→ training/validation data
```

1. **Vocabulary Building**: A bidirectional mapping (`token_to_id` and `id_to_token`) is deterministically constructed by ranking tokens by frequency (descending) with alphabetical tie-breaking. Every unique word is mapped to an integer index `[0, vocab_size - 1]`.
2. **Integer Encoding**: The preprocessed token stream is converted into an array of integer IDs.
3. **Fixed-Length Sequences & Next-Token Targets**: A sliding window of length `sequence_length` (default: 20) moves across the token sequence. For each window, the input `X` is the sequence of 20 token IDs, and the target `y` is the immediate next token ID.
4. **Why the Target is the Next Token**: Generative language modeling is fundamentally autoregressive: given a sequence of preceding context words $(w_1, w_2, \dots, w_t)$, the model learns the conditional probability distribution $P(w_{t+1} \mid w_1, \dots, w_t)$. During text generation, the model predicts the next word, appends it to the prompt, and shifts the context window forward step-by-step.
5. **Chronological Train/Validation Split**: The token corpus is split chronologically before creating training and validation sequences, preventing overlapping sliding-window sequences from crossing the train-validation boundary.
6. **Vocabulary Persistence**: The vocabulary mappings are saved in JSON format under `models/token_to_id.json` and `models/id_to_token.json` for reuse during model inference.

### Running Task 2 Sequence Preparation

To build the vocabulary, encode sequences, and save vocabulary files:

```bash
python src/prepare_data.py
```

### Dataset Preparation Statistics

Running Task 2 on the preprocessed Shakespeare corpus yields:

| Metric | Value |
| :--- | :--- |
| **Total tokens** | 966,377 |
| **Vocabulary size** | 30,536 |
| **Sequence length** | 20 |
| **Total sequences (chronological split)** | 966,337 |
| **Training sequences (80%)** | 773,081 |
| **Validation sequences (20%)** | 193,256 |
| **X_train shape** | `(773081, 20)` |
| **y_train shape** | `(773081,)` |
| **X_val shape** | `(193256, 20)` |
| **y_val shape** | `(193256,)` |


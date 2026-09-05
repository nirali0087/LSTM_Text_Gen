# LSTM Text Generator

## Objective
A generative text generation system using an LSTM recurrent neural network to generate Shakespeare-style text based on seed prompts.

## Dataset
William Shakespeare's complete works obtained from [Project Gutenberg](https://www.gutenberg.org/) (`data/shakespeare.txt`).

---

## Task 1: Text Preprocessing
Pipeline implemented in `src/preprocess.py`:
- Raw text → Lowercase → Remove punctuation → Normalize whitespace → Word tokenization

Run:
```bash
python src/preprocess.py
```

- Total tokens: `966,377`
- Unique tokens (vocabulary): `30,536`

---

## Task 2: Sequence Preparation
Pipeline implemented in `src/prepare_data.py`:
- Deterministic vocabulary mapping (`token_to_id.json` & `id_to_token.json` in `models/`)
- Integer encoding of tokens
- Sliding window sequences (`sequence_length = 20`) for next-token prediction
- Chronological train/validation split (80% / 20%)

Run:
```bash
python src/prepare_data.py
```

- Training sequences: `773,081` (shape: `(773081, 20)`)
- Validation sequences: `193,256` (shape: `(193256, 20)`)

---

## Task 3: LSTM Model Architecture
Model architecture implemented in `src/model.py`:

```text
Input (shape: (20,), dtype: int32)
  ↓
Embedding (input_dim: 30,536, output_dim: 128)
  ↓
LSTM (units: 256, return_sequences: False)
  ↓
Dropout (rate: 0.2)
  ↓
Dense (units: 30,536, activation: softmax)
```

- **Optimizer**: Adam (learning rate: `0.001`)
- **Loss**: `SparseCategoricalCrossentropy(from_logits=False)`
- **Metrics**: `accuracy`
- **Total Parameters**: `12,150,600` (all trainable)

Run architecture validation:
```bash
python src/model.py
```

*(Note: Task 3 validates the model architecture and forward pass only; the model is not trained yet.)*

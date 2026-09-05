# LSTM Text Generator

A generative text generation system built with an LSTM neural network to learn word-level language patterns from Shakespeare's complete works and generate new text from a user-provided seed sequence.

---

## Project Overview

This project implements an end-to-end LSTM-based text generation pipeline:

```text
Shakespeare Dataset
       ↓
Text Preprocessing
       ↓
Vocabulary Construction
       ↓
Integer Encoding
       ↓
Input Sequence + Next-Token Target
       ↓
LSTM Neural Network
       ↓
Next-Token Probability Distribution
       ↓
Temperature-Based Sampling
       ↓
Generated Text
```

The implementation uses Python, TensorFlow/Keras, NumPy, and standard Python libraries.

---

## Dataset

The model uses the complete works of William Shakespeare obtained from Project Gutenberg.

- **Dataset source**: [https://www.gutenberg.org/]
- **Dataset location**: `data/shakespeare.txt`

The original dataset file is preserved unchanged.

---

## Task 1 — Text Preprocessing

Implemented in: `src/preprocess.py`

### Preprocessing Pipeline
```text
Raw Text → Lowercase → Remove Punctuation → Normalize Whitespace → Word Tokenization
```

### Dataset Statistics
| Metric | Value |
| :--- | :--- |
| **Total tokens** | 966,377 |
| **Unique tokens (Vocabulary)** | 30,536 |

**Run preprocessing:**
```bash
python src/preprocess.py
```

---

## Task 2 — Vocabulary & Sequence Preparation

Implemented in: `src/prepare_data.py`

The preprocessing output is converted into integer token IDs and prepared for next-token prediction.

### Main Components
- Deterministic vocabulary construction
- Token-to-ID mapping
- ID-to-token mapping
- Integer encoding
- Sliding-window sequence generation
- Next-token target creation
- Chronological train/validation split

**Vocabulary files:**
- `models/token_to_id.json`
- `models/id_to_token.json`

### Sequence Configuration
- **Sequence length**: 20

The prepared corpus contains:
| Split | Sequences |
| :--- | :--- |
| **Training sequences** | 773,081 |
| **Validation sequences** | 193,256 |

> The split is performed chronologically, and sequences are created independently within each split so that a sliding-window sequence does not cross the train/validation boundary.

**Run sequence preparation:**
```bash
python src/prepare_data.py
```

---

## Task 3 — LSTM Model Architecture

Implemented in: `src/model.py`

### Architecture
```text
Input (sequence_length = 20)
       ↓
Embedding (vocabulary = 30,536, dimension = 128)
       ↓
LSTM (256 units)
       ↓
Dropout (rate = 0.2)
       ↓
Dense (30,536 units)
       ↓
Softmax
```

### Model Configuration
| Parameter | Value |
| :--- | :--- |
| **Vocabulary Size** | 30,536 |
| **Sequence Length** | 20 |
| **Embedding Dimension** | 128 |
| **LSTM Units** | 256 |
| **Dropout** | 0.2 |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Loss** | Sparse Categorical Crossentropy |
| **Metric** | Accuracy |
| **Total Parameters** | 12,150,600 |

The output layer produces a probability distribution over the complete vocabulary for next-token prediction.

### Architecture Validation
```bash
python src/model.py
```
*The architecture was successfully validated using real input sequences.*

---

## Task 4 — Model Training

The training pipeline is implemented in: `src/model.py`

### Training Configuration
| Parameter | Configuration |
| :--- | :--- |
| **Batch Size** | 128 |
| **Maximum Epochs** | 10 |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Loss** | SparseCategoricalCrossentropy |
| **Early Stopping** | Yes (`patience = 2`, `restore_best_weights = True`) |
| **Model Checkpointing** | Yes (`models/best_model.keras`) |
| **Learning Rate Scheduling** | ReduceLROnPlateau (`factor = 0.5`, `patience = 1`) |

### Training Strategy
The complete preprocessing and sequence-generation pipeline supports the full prepared dataset.

For the practical training run, a representative chronological subset was used because the 30,536-class softmax output makes CPU-based training computationally intensive.

**Training subset:**
| Split | Sequences |
| :--- | :--- |
| **Training sequences** | 10,000 |
| **Validation sequences** | 2,500 |

This preserves the same sequence format and chronological train/validation methodology used by the complete pipeline.

### Actual Training Results
The training run completed with EarlyStopping after 4 epochs.

| Metric | Result |
| :--- | :--- |
| **Final Training Loss** | 6.3488 |
| **Final Training Accuracy** | 0.0312 |
| **Final Validation Loss** | 8.0061 |
| **Final Validation Accuracy** | 0.0200 |
| **Best Validation Loss** | 7.8778 |
| **Best Validation Accuracy** | 0.0148 |

The best checkpoint was obtained at **Epoch 2**.

- **Saved Model**: `models/best_model.keras`
- **Training History**: `models/training_history.json`

The trained model was successfully saved, reloaded, and evaluated.

### Computational Note
The model contains approximately 12.15 million parameters, including a large vocabulary projection layer.

Because of this architecture, full-dataset CPU training requires substantial computation time. The representative subset was therefore used for the practical training run while keeping the complete data-preparation pipeline available.

---

## Task 5 — Text Generation

Implemented in: `src/generate.py`

The generator loads the trained model and vocabulary and predicts the next word iteratively.

### Generation Process
```text
Seed Text → Tokenization → Token IDs → Latest 20 Tokens → LSTM Prediction → Temperature Sampling → Next Token → Repeat → Generated Text
```

Temperature sampling allows control over generation diversity:
- **Lower temperature** → more focused/deterministic output
- **Higher temperature** → more diverse/random output

**Run generation:**
```bash
python src/generate.py
```

### Generated Examples
The following examples were generated directly by the trained model:

#### Example 1
- **Seed**: `to be or not to be`
- **Temperature**: `0.7`
- **Generated text**:
  > *to be or not to be nor i my be parts to when why by the the to be may is and do horse doth praise it that but and chest his made love praise the in you their made me even your when in i*

#### Example 2
- **Seed**: `shall i compare thee`
- **Temperature**: `0.7`
- **Generated text**:
  > *shall i compare thee to me but thee to the upon a they keeps as me but in a shall i of to to the of then and as and to or than doth chide all to my the the my in and all*

#### Example 3
- **Seed**: `all the world's a stage`
- **Temperature**: `0.7`
- **Generated text**:
  > *all the world's a stage art when thy my think for and so or or thou i my i for must the against in to place not that eyelids to dear they if to as thee a the good all you in times the thee*

Generation was successfully executed for all three seed prompts.


## Project Structure

```text
LSTM_Text_Gen/
├── data/
│   └── shakespeare.txt
├── models/
│   ├── token_to_id.json
│   ├── id_to_token.json
│   ├── best_model.keras
│   └── training_history.json
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── prepare_data.py
│   ├── model.py
│   └── generate.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Preprocess Dataset
```bash
python src/preprocess.py
```

### 2. Prepare Sequences
```bash
python src/prepare_data.py
```

### 3. Validate Model Architecture
```bash
python src/model.py
```

### 4. Train Model
```bash
python src/model.py --train
```

### 5. Generate Text
Ensure that the trained model is available locally: `models/best_model.keras`
```bash
python src/generate.py
```

---

## Model File Note

The trained model file:
`models/best_model.keras`
is approximately **145.8 MB**.

Because GitHub limits individual files to 100 MB, the `.keras` model file is excluded from the Git repository using `.gitignore`.

The vocabulary files and source code are included in the repository. The trained model file is required locally to reproduce the generated text.

---

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **LSTM / Recurrent Neural Networks**
- **Natural Language Processing (NLP)**
- **Word-level Tokenization**
- **Temperature-based Sampling**


## Final Validation

- Dataset preprocessing completed successfully.
- Vocabulary and sequence preparation completed successfully.
- LSTM architecture validated successfully.
- Model training completed successfully on the documented representative subset.
- Best model checkpoint saved and successfully reloaded.
- Validation evaluation completed successfully.
- Text generation successfully tested with multiple seed prompts.
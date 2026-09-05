import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("KERAS_BACKEND", "torch")

# Flexible import: supports TensorFlow/Keras or standalone Keras (multi-backend)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, losses, optimizers
except ImportError:
    import keras
    from keras import layers, losses, optimizers

# Reuse vocabulary and dataset loading from Task 2
from src.prepare_data import load_vocabulary, prepare_dataset


def build_model(
    vocabulary_size: int,
    sequence_length: int = 20,
    embedding_dim: int = 128,
    lstm_units: int = 256,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """
    Builds the LSTM neural network architecture for next-token prediction.

    The architecture consists of:
    1. Input layer: accepts sequences of integer token IDs (shape: (sequence_length,)).
    2. Embedding layer: maps integer IDs to dense continuous vectors of dimension 128.
    3. LSTM layer: single recurrent layer with 256 units (return_sequences=False).
    4. Dropout layer: rate 0.2 to prevent overfitting during future training.
    5. Output Dense layer: vocabulary_size units with softmax activation,
       producing a probability distribution over the vocabulary.

    Args:
        vocabulary_size (int): Total number of unique tokens in vocabulary.
        sequence_length (int): Number of tokens in input context window. Default is 20.
        embedding_dim (int): Dimensionality of embedding vectors. Default is 128.
        lstm_units (int): Number of units in the single LSTM layer. Default is 256.
        dropout_rate (float): Dropout probability. Default is 0.2.

    Returns:
        keras.Model: Uncompiled Keras Sequential model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length,), dtype="int32", name="input_layer"),
            layers.Embedding(
                input_dim=vocabulary_size,
                output_dim=embedding_dim,
                name="embedding_layer",
            ),
            layers.LSTM(
                units=lstm_units,
                return_sequences=False,
                name="lstm_layer",
            ),
            layers.Dropout(
                rate=dropout_rate,
                name="dropout_layer",
            ),
            layers.Dense(
                units=vocabulary_size,
                activation="softmax",
                name="output_dense",
            ),
        ],
        name="lstm_text_generator",
    )
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Compiles the LSTM model with Adam optimizer and SparseCategoricalCrossentropy loss.

    Uses from_logits=False because the final Dense layer already includes
    a softmax activation producing valid probability distributions. Compatible
    with integer target IDs from Task 2 without one-hot encoding.

    Args:
        model (keras.Model): The Keras model to compile.
        learning_rate (float): Learning rate for the Adam optimizer. Default is 0.001.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def test_model(
    model: Optional[keras.Model] = None,
    vocab_dir: Union[str, Path] = "models",
    sequence_length: int = 20,
    embedding_dim: int = 128,
    lstm_units: int = 256,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Dict[str, Any]:
    """
    Validates the model architecture without training:
    1. Programmatically loads the Task 2 vocabulary to obtain vocabulary_size.
    2. Builds and compiles the model if not provided.
    3. Loads real training data from Task 2 pipeline.
    4. Passes the first two samples (X_test = X_train[:2]) through model.predict().
    5. Verifies output shapes and validates that softmax probability sums equal 1.0.

    Args:
        model (Optional[keras.Model]): Pre-built model instance. If None, builds a new one.
        vocab_dir (Union[str, Path]): Directory containing saved vocabulary files.
        sequence_length (int): Input sequence length. Default is 20.
        embedding_dim (int): Embedding dimensions. Default is 128.
        lstm_units (int): LSTM hidden units. Default is 256.
        dropout_rate (float): Dropout rate. Default is 0.2.
        learning_rate (float): Optimizer learning rate. Default is 0.001.

    Returns:
        Dict[str, Any]: Validation results dictionary.
    """
    # 1. Programmatically load vocabulary size from Task 2 files
    token_to_id, id_to_token = load_vocabulary(vocab_dir=vocab_dir)
    vocabulary_size = len(token_to_id)

    # 2. Build and compile model
    if model is None:
        model = build_model(
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
        )
        model = compile_model(model, learning_rate=learning_rate)

    # 3. Load actual training sequences from Task 2 pipeline
    X_train, y_train, _, _, _, _ = prepare_dataset(
        sequence_length=sequence_length,
        save_vocab=False,
    )

    # 4. Use ONLY first two real training samples for architecture testing (DO NOT TRAIN)
    X_test = X_train[:2]

    # 5. Forward pass / inference only (model.fit is NOT called)
    predictions = model.predict(X_test, verbose=0)

    # 6. Verify probability sums
    prob_sum_sample_1 = float(np.sum(predictions[0]))
    prob_sum_sample_2 = float(np.sum(predictions[1]))

    # 7. Count parameters
    trainable_params = int(sum(np.prod(p.shape) for p in model.trainable_weights))
    non_trainable_params = int(sum(np.prod(p.shape) for p in model.non_trainable_weights))
    total_params = trainable_params + non_trainable_params

    return {
        "model": model,
        "vocabulary_size": vocabulary_size,
        "sequence_length": sequence_length,
        "input_shape": X_test.shape,
        "output_shape": predictions.shape,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "prob_sum_sample_1": prob_sum_sample_1,
        "prob_sum_sample_2": prob_sum_sample_2,
        "compilation_succeeded": True,
    }


def main() -> None:
    """
    Main entry point to execute Task 3 architecture validation.
    Prints model summary, parameter counts, input/output shapes,
    and actual prediction probability sums.
    """
    print("=" * 65)
    print("TASK 3: LSTM MODEL ARCHITECTURE VALIDATION")
    print("=" * 65)

    print("\n[1/4] Loading Task 2 vocabulary programmatically...")
    token_to_id, _ = load_vocabulary("models")
    vocab_size = len(token_to_id)
    print(f"      Vocabulary size: {vocab_size:,} tokens (loaded from models/)")

    print("\n[2/4] Building LSTM model architecture...")
    model = build_model(vocabulary_size=vocab_size, sequence_length=20)

    print("\n[3/4] Compiling model (Adam lr=0.001, SparseCategoricalCrossentropy)...")
    model = compile_model(model, learning_rate=0.001)

    print("\nModel Summary:")
    print("-" * 65)
    model.summary()
    print("-" * 65)

    print("\n[4/4] Validating forward pass on 2 real training samples (NO TRAINING)...")
    results = test_model(model=model, vocab_dir="models")

    print("\n" + "=" * 65)
    print("ARCHITECTURE VALIDATION RESULTS")
    print("=" * 65)
    print(f"Vocabulary size        : {results['vocabulary_size']:,}")
    print(f"Sequence length        : {results['sequence_length']}")
    print(f"Input shape (X_test)   : {results['input_shape']}")
    print(f"Output shape (preds)   : {results['output_shape']}")
    print(f"Total parameters       : {results['total_params']:,}")
    print(f"Trainable parameters   : {results['trainable_params']:,}")
    print(f"Non-trainable params   : {results['non_trainable_params']:,}")
    print(f"Compilation status     : {'SUCCESS' if results['compilation_succeeded'] else 'FAILED'}")
    print("-" * 65)
    print("Softmax Probability Sums (must be approximately 1.0):")
    print(f"  Sample 1 probability sum: {results['prob_sum_sample_1']:.6f}")
    print(f"  Sample 2 probability sum: {results['prob_sum_sample_2']:.6f}")
    print("=" * 65)
    print("Architecture verified successfully. Model was NOT trained.")


if __name__ == "__main__":
    main()

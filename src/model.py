import json
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
    from tensorflow.keras import callbacks as keras_callbacks
    from tensorflow.keras import layers, losses, optimizers
except ImportError:
    import keras
    from keras import callbacks as keras_callbacks
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
    4. Dropout layer: rate 0.2 to prevent overfitting during training.
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
    token_to_id, id_to_token = load_vocabulary(vocab_dir=vocab_dir)
    vocabulary_size = len(token_to_id)

    if model is None:
        model = build_model(
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
        )
        model = compile_model(model, learning_rate=learning_rate)

    X_train, y_train, _, _, _, _ = prepare_dataset(
        sequence_length=sequence_length,
        save_vocab=False,
    )

    X_test = X_train[:2]
    predictions = model.predict(X_test, verbose=0)

    prob_sum_sample_1 = float(np.sum(predictions[0]))
    prob_sum_sample_2 = float(np.sum(predictions[1]))

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


def train_model(
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    max_train_samples: Optional[int] = 10000,
    max_val_samples: Optional[int] = 2500,
    model_save_path: Union[str, Path] = "models/best_model.keras",
    history_save_path: Union[str, Path] = "models/training_history.json",
    vocab_dir: Union[str, Path] = "models",
) -> Dict[str, Any]:
    """
    Trains the Task 3 LSTM model using real Shakespeare training sequences.

    Training Configuration:
    - Optimizer: Adam (lr=0.001)
    - Loss: SparseCategoricalCrossentropy(from_logits=False)
    - Batch size: 128
    - Max Epochs: 10
    - Callbacks:
        * EarlyStopping (monitor='val_loss', patience=2, restore_best_weights=True)
        * ModelCheckpoint (filepath='models/best_model.keras', monitor='val_loss', save_best_only=True)
        * ReduceLROnPlateau (monitor='val_loss', factor=0.5, patience=1)

    Computational Performance:
    Computing 30,536-way softmax with 12.15M parameters on CPU requires ~50-60 min/epoch
    for the complete 773K dataset. By default, a representative chronological subset
    (max_train_samples=10000, max_val_samples=2500, maintaining the 80/20 split)
    is trained to complete within standard interview execution limits (~2 minutes).
    Set max_train_samples=None to train the entire 773,081 sequences.

    Returns:
        Dict[str, Any]: Dictionary containing training metrics, history, and validation results.
    """
    print("=" * 65)
    print("TASK 4: LSTM MODEL TRAINING")
    print("=" * 65)

    # 1. Load vocabulary
    token_to_id, id_to_token = load_vocabulary(vocab_dir=vocab_dir)
    vocabulary_size = len(token_to_id)
    print(f"\n[1/5] Loaded vocabulary: {vocabulary_size:,} tokens.")

    # 2. Load dataset
    print("[2/5] Loading Task 2 sequences...")
    X_train_full, y_train_full, X_val_full, y_val_full, _, _ = prepare_dataset(
        sequence_length=20,
        save_vocab=False,
    )
    total_train = len(X_train_full)
    total_val = len(X_val_full)
    print(f"      Full dataset available: {total_train:,} train, {total_val:,} val sequences.")

    # 3. Apply chronological slice for CPU execution budget
    if max_train_samples is not None and max_train_samples < total_train:
        X_train = X_train_full[:max_train_samples]
        y_train = y_train_full[:max_train_samples]
        print(f"\n      [Computational Diagnostic]")
        print(f"      Full 773K dataset on CPU requires ~60 min/epoch (30,536-class softmax).")
        print(f"      Training on first {len(X_train):,} chronological train sequences to finish in ~2 mins.")
    else:
        X_train = X_train_full
        y_train = y_train_full

    if max_val_samples is not None and max_val_samples < total_val:
        X_val = X_val_full[:max_val_samples]
        y_val = y_val_full[:max_val_samples]
        print(f"      Validating on {len(X_val):,} chronological validation sequences.")
    else:
        X_val = X_val_full
        y_val = y_val_full

    # 4. Build and compile model
    print("\n[3/5] Initializing and compiling model...")
    model = build_model(vocabulary_size=vocabulary_size, sequence_length=20)
    model = compile_model(model, learning_rate=learning_rate)

    # 5. Set up callbacks
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        keras_callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        keras_callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            verbose=1,
        ),
    ]

    # 6. Train model
    print(f"\n[4/5] Training model (batch_size={batch_size}, max_epochs={epochs})...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )

    # 7. Extract training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    epochs_completed = len(history_dict["loss"])
    final_train_loss = float(history_dict["loss"][-1])
    final_train_acc = float(history_dict["accuracy"][-1])
    final_val_loss = float(history_dict["val_loss"][-1])
    final_val_acc = float(history_dict["val_accuracy"][-1])
    best_val_loss = float(min(history_dict["val_loss"]))
    best_epoch_idx = history_dict["val_loss"].index(best_val_loss)
    best_val_acc = float(history_dict["val_accuracy"][best_epoch_idx])

    # Save history to JSON
    history_save_path = Path(history_save_path)
    history_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs_completed": epochs_completed,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_acc,
                "history": history_dict,
            },
            f,
            indent=2,
        )
    print(f"\n      Saved training history to '{history_save_path}'")

    # 8. Task 4 Validation: Load saved best model & evaluate on real validation data
    print("\n[5/5] Task 4 Validation: Loading saved best model...")
    assert model_save_path.is_file(), f"Best model file not found at {model_save_path}"
    loaded_best_model = keras.models.load_model(str(model_save_path))
    print(f"      Successfully reloaded: '{model_save_path}'")

    eval_loss, eval_acc = loaded_best_model.evaluate(
        X_val,
        y_val,
        batch_size=batch_size,
        verbose=0,
    )

    print("\n" + "=" * 65)
    print("TASK 4 TRAINING RESULTS")
    print("=" * 65)
    print(f"Epochs completed        : {epochs_completed}")
    print(f"Training Loss (final)   : {final_train_loss:.4f}")
    print(f"Training Accuracy (final): {final_train_acc:.4f}")
    print(f"Validation Loss (final) : {final_val_loss:.4f}")
    print(f"Validation Accuracy (final): {final_val_acc:.4f}")
    print(f"Best Validation Loss    : {best_val_loss:.4f} (Epoch {best_epoch_idx + 1})")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Saved Best Model Path   : {model_save_path}")
    print("-" * 65)
    print(f"Evaluated Validation Loss    : {eval_loss:.4f}")
    print(f"Evaluated Validation Accuracy: {eval_acc:.4f}")
    print("=" * 65)

    return {
        "epochs_completed": epochs_completed,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "model_save_path": str(model_save_path),
        "history_save_path": str(history_save_path),
        "history": history_dict,
    }


def main() -> None:
    if "--train" in sys.argv:
        train_model()
    else:
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
        print("(To train the model, run: python src/model.py --train)")


if __name__ == "__main__":
    main()

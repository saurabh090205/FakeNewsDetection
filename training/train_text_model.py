import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

try:
    from data_pipeline import METRICS_DIR, PROCESSED_DIR, prepare_dataset
    from settings import BATCH_SIZE, DENSE_UNITS, DROPOUT, EMBEDDING_DIMENSION, EPOCHS, MAX_VOCABULARY, RECURRENT_UNITS, ROOT, SEQUENCE_LENGTH, SEED
except ModuleNotFoundError:
    from training.data_pipeline import METRICS_DIR, PROCESSED_DIR, prepare_dataset
    from training.settings import BATCH_SIZE, DENSE_UNITS, DROPOUT, EMBEDDING_DIMENSION, EPOCHS, MAX_VOCABULARY, RECURRENT_UNITS, ROOT, SEQUENCE_LENGTH, SEED


def train(model_name: str):
    if model_name not in {"dense", "rnn", "lstm", "gru"}:
        raise ValueError("Supported model names: dense, rnn, lstm, gru")
    data_path = PROCESSED_DIR / "news.csv"
    if not data_path.is_file():
        prepare_dataset()
    data = pd.read_csv(data_path)
    with (PROCESSED_DIR / "label_mapping.json").open(encoding="utf-8") as file:
        mapping = json.load(file)
    random.seed(SEED)
    np.random.seed(SEED)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)

    train_rows = data[data.split == "train"]
    val_rows = data[data.split == "validation"]
    test_rows = data[data.split == "test"]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCABULARY, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_rows.text.tolist())
    sequences = tokenizer.texts_to_sequences(data.text.tolist())
    X = tf.keras.utils.pad_sequences(sequences, maxlen=SEQUENCE_LENGTH, padding="post", truncating="post")
    y = data.label_id.to_numpy()
    train_x, val_x, test_x = X[train_rows.index], X[val_rows.index], X[test_rows.index]
    train_y, val_y, test_y = y[train_rows.index], y[val_rows.index], y[test_rows.index]

    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH,))
    x = tf.keras.layers.Embedding(MAX_VOCABULARY, EMBEDDING_DIMENSION, mask_zero=True)(inputs)
    if model_name == "dense":
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
    elif model_name == "rnn":
        x = tf.keras.layers.SimpleRNN(RECURRENT_UNITS)(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
    elif model_name == "lstm":
        x = tf.keras.layers.LSTM(RECURRENT_UNITS)(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
    else:
        x = tf.keras.layers.GRU(RECURRENT_UNITS)(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model_dir = ROOT / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(model_dir / "model.keras", monitor="val_loss", save_best_only=True)]
    started = time.perf_counter()
    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
    duration = time.perf_counter() - started
    model = tf.keras.models.load_model(model_dir / "model.keras")
    probs = model.predict(test_x, verbose=0).reshape(-1)
    predictions = (probs >= 0.5).astype(int)
    y_true = test_y.astype(int)
    if len(np.unique(y_true)) < 2 or len(np.unique(predictions)) < 2 or min(np.bincount(predictions, minlength=2)) < max(10, math.ceil(len(predictions) * 0.05)):
        raise RuntimeError(f"{model_name} appears degenerate on the untouched test set; inspect data, labels, and preprocessing before continuing.")
    test_table = test_rows[["headline", "article", "label"]].copy()
    test_table["predicted"] = [mapping["index_to_class"][str(int(pred))] for pred in predictions]
    test_table["fake_probability"] = np.where(mapping["sigmoid_positive_label"] == "FAKE", probs, 1 - probs)
    test_table["real_probability"] = 1 - test_table.fake_probability
    validation_samples = pd.concat([
        test_table[test_table.label == label].head(10) for label in ("FAKE", "REAL")
    ])
    if validation_samples.label.value_counts().min() < 10:
        raise RuntimeError("Test split needs at least 10 REAL and 10 FAKE samples for mandatory model validation.")
    print("Actual | Predicted | FAKE prob | REAL prob")
    for row in validation_samples.itertuples():
        print(f"{row.label} | {row.predicted} | {row.fake_probability:.4f} | {row.real_probability:.4f}")
    results = {
        "model": model_name, "description": f"{model_name.upper()} neural text classifier",
        "parameter_count": int(model.count_params()), "training_seconds": duration,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "history": history.history,
        "test_accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "hyperparameters": {"vocabulary_size": MAX_VOCABULARY, "sequence_length": SEQUENCE_LENGTH, "embedding_dimension": EMBEDDING_DIMENSION, "dense_units": DENSE_UNITS, "recurrent_units": RECURRENT_UNITS, "dropout": DROPOUT, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "seed": SEED},
    }
    with (model_dir / "tokenizer.json").open("w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
    with (METRICS_DIR / f"{model_name}.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    test_table.to_csv(METRICS_DIR / f"{model_name}_test_samples.csv", index=False)
    refresh_comparison()
    print(json.dumps({key: results[key] for key in ("test_accuracy", "precision", "recall", "f1", "roc_auc")}, indent=2))


def refresh_comparison():
    metrics = {}
    for path in METRICS_DIR.glob("*.json"):
        if path.name in {"dataset_stats.json", "comparison.json"}:
            continue
        with path.open(encoding="utf-8") as f:
            item = json.load(f)
        if "test_accuracy" in item:
            metrics[item["model"]] = item
    comparison = {}
    for key in ("test_accuracy", "precision", "recall", "f1"):
        comparison[key] = max(metrics, key=lambda name: metrics[name][key]) if metrics else None
    with (METRICS_DIR / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump({"best_models": comparison, "models": {name: {key: value[key] for key in ("test_accuracy", "precision", "recall", "f1")} for name, value in metrics.items()}}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("dense", "rnn", "lstm", "gru"))
    train(parser.parse_args().model)

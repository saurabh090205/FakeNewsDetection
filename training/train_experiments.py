"""Run explicit overfit and underfit demonstrations on train/validation only."""
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

try:
    from data_pipeline import METRICS_DIR, PROCESSED_DIR, prepare_dataset
    from settings import BATCH_SIZE, MAX_VOCABULARY, SEQUENCE_LENGTH
except ModuleNotFoundError:
    from training.data_pipeline import METRICS_DIR, PROCESSED_DIR, prepare_dataset
    from training.settings import BATCH_SIZE, MAX_VOCABULARY, SEQUENCE_LENGTH


def main():
    if not (PROCESSED_DIR / "news.csv").is_file():
        prepare_dataset()
    data = pd.read_csv(PROCESSED_DIR / "news.csv")
    train = data[data.split == "train"]
    validation = data[data.split == "validation"]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCABULARY, oov_token="<OOV>")
    tokenizer.fit_on_texts(train.text.tolist())
    sequences = tokenizer.texts_to_sequences(data.text.tolist())
    padded = tf.keras.utils.pad_sequences(sequences, maxlen=SEQUENCE_LENGTH, padding="post", truncating="post")
    train_x, val_x = padded[train.index], padded[validation.index]
    train_y, val_y = data.label_id.to_numpy()[train.index], data.label_id.to_numpy()[validation.index]
    results = {}
    for name, embedding_dim, units, epochs, dropout in (
        ("overfit", 128, 128, 20, 0.0), ("underfit", 4, 2, 1, 0.0),
    ):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,)),
            tf.keras.layers.Embedding(MAX_VOCABULARY, embedding_dim, mask_zero=True),
            tf.keras.layers.LSTM(units),
            *([tf.keras.layers.Dropout(dropout)] if dropout else []),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, batch_size=BATCH_SIZE, verbose=1)
        results[name] = history.history
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "experiment_curves.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()

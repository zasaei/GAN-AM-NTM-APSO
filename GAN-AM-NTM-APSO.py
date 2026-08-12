import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from GAN import EnvironmentalGAN
from NTM import build_am_ntm
from APSO import optimize_hyperparameters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_FILE = "DataSet.xlsx"
TARGETS = ["PM2.5", "PM10", "AQI"]

SEQ_LEN = 24
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

GAN_EPOCHS = 300
FINAL_EPOCHS = 50
BATCH_SIZE = 64


def load_dataset():
    df = (
        pd.read_excel(DATA_FILE)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    missing = [
        c for c in TARGETS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing target columns: {missing}"
        )

    numeric = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    features = [
        c for c in numeric
        if c not in TARGETS
    ]

    if not features:
        raise ValueError(
            "No predictor columns were found."
        )

    return df, features


def make_sequences(X, y):
    Xs, ys = [], []

    for i in range(
        len(X) - SEQ_LEN
    ):
        Xs.append(
            X[i:i + SEQ_LEN]
        )
        ys.append(
            y[i + SEQ_LEN]
        )

    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32)
    )


def evaluate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(
            y_true, y_pred
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true, y_pred
            )
        ),
        "MAPE (%)": np.mean(
            np.abs(
                (y_true - y_pred)
                / np.maximum(
                    np.abs(y_true),
                    1e-8
                )
            )
        ) * 100,
        "R2": r2_score(
            y_true, y_pred
        )
    }


def main():

    df, features = load_dataset()

    columns = features + TARGETS

    raw = df[columns].values.astype(
        np.float32
    )

    n = len(raw)

    train_end = int(
        TRAIN_RATIO * n
    )

    val_end = int(
        (TRAIN_RATIO + VAL_RATIO) * n
    )

    train_raw = raw[:train_end]
    val_raw = raw[train_end:val_end]
    test_raw = raw[val_end:]

    scaler = MinMaxScaler(
        feature_range=(-1, 1)
    )

    train = scaler.fit_transform(
        train_raw
    ).astype(np.float32)

    val = scaler.transform(
        val_raw
    ).astype(np.float32)

    test = scaler.transform(
        test_raw
    ).astype(np.float32)

    # -------------------------------------------------------
    # GAN augmentation is applied ONLY to the training data.
    # -------------------------------------------------------

    gan = EnvironmentalGAN(
        n_features=train.shape[1]
    )

    gan_batch = min(
        BATCH_SIZE,
        len(train)
    )

    gan.train(
        train,
        epochs=GAN_EPOCHS,
        batch_size=gan_batch
    )

    synthetic_count = max(
        1,
        int(0.30 * len(train))
    )

    synthetic = gan.generate(
        synthetic_count
    )

    augmented_train = np.vstack(
        [train, synthetic]
    )

    n_features = len(features)
    n_targets = len(TARGETS)

    X_train, y_train = make_sequences(
        augmented_train[:, :],
        augmented_train[:, n_features:]
    )

    X_val, y_val = make_sequences(
        val[:, :],
        val[:, n_features:]
    )

    X_test, y_test = make_sequences(
        test[:, :],
        test[:, n_features:]
    )

    # AM-NTM receives predictor variables only.
    X_train = X_train[:, :, :n_features]
    X_val = X_val[:, :, :n_features]
    X_test = X_test[:, :, :n_features]

    # -------------------------------------------------------
    # APSO objective
    # -------------------------------------------------------

    def objective(position):

        controller_size = int(
            round(position[0])
        )

        memory_size = int(
            round(position[1])
        )

        learning_rate = float(
            position[2]
        )

        model = build_am_ntm(
            input_dim=n_features,
            output_dim=n_targets,
            controller_size=controller_size,
            memory_size=memory_size,
            learning_rate=learning_rate
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(
                X_val,
                y_val
            ),
            epochs=3,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        return float(
            model.evaluate(
                X_val,
                y_val,
                verbose=0
            )[0]
        )

    bounds = [
        (64, 192),
        (32, 96),
        (1e-4, 3e-3)
    ]

    best_position, best_score, history = (
        optimize_hyperparameters(
            objective,
            bounds,
            n_particles=6,
            max_iter=5,
            seed=SEED,
            verbose=True
        )
    )

    controller_size = int(
        round(best_position[0])
    )

    memory_size = int(
        round(best_position[1])
    )

    learning_rate = float(
        best_position[2]
    )

    print(
        "\nBest APSO parameters:"
    )

    print(
        "controller_size =",
        controller_size
    )

    print(
        "memory_size =",
        memory_size
    )

    print(
        "learning_rate =",
        learning_rate
    )

    # -------------------------------------------------------
    # Final AM-NTM model
    # -------------------------------------------------------

    model = build_am_ntm(
        input_dim=n_features,
        output_dim=n_targets,
        controller_size=controller_size,
        memory_size=memory_size,
        learning_rate=learning_rate
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=FINAL_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    prediction_scaled = model.predict(
        X_test,
        verbose=0
    )

    # Inverse-transform only the target dimensions.
    target_scaler = MinMaxScaler(
        feature_range=(-1, 1)
    )

    target_scaler.fit(
        train_raw[:, n_features:]
    )

    prediction = target_scaler.inverse_transform(
        prediction_scaled[:, -1, :]
    )

    true = target_scaler.inverse_transform(
        y_test
    )

    results = []

    for i, target in enumerate(
        TARGETS
    ):

        result = evaluate_metrics(
            true[:, i],
            prediction[:, i]
        )

        results.append(
            {
                "Target": target,
                **result
            }
        )

        print(
            f"\n{target}:"
        )

        for key, value in result.items():
            print(
                f"{key}: {value:.6f}"
            )

    pd.DataFrame(
        results
    ).to_csv(
        "GAN_AM_NTM_APSO_results.csv",
        index=False
    )

    pd.DataFrame(
        history
    ).to_csv(
        "APSO_history.csv",
        index=False
    )

    model.save(
        "GAN_AM_NTM_APSO_model.keras"
    )

    print(
        "\nExecution completed."
    )


if __name__ == "__main__":
    main()

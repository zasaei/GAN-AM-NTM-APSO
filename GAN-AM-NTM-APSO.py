"""
GAN-AM-NTM-APSO.py

Main experimental pipeline:
DataSet.xlsx
    ↓
Preprocessing
    ↓
GAN augmentation
    ↓
AM-NTM temporal learning
    ↓
APSO optimization
    ↓
PM2.5 / PM10 / AQI prediction
    ↓
MAE / RMSE / MAPE / R2
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Local research modules
from GAN import generate_synthetic_data
from NTM import build_am_ntm_model
from APSO import optimize_model


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. Configuration
# ============================================================

DATA_FILE = "DataSet.xlsx"

TARGETS = [
    "PM2.5",
    "PM10",
    "AQI"
]

SEQUENCE_LENGTH = 24

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 64
EPOCHS = 50


# ============================================================
# 3. Load dataset
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_excel(DATA_FILE)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset columns:")
    print(df.columns.tolist())

    return df


# ============================================================
# 4. Prepare features and targets
# ============================================================

def prepare_data(df):

    missing_targets = [
        target for target in TARGETS
        if target not in df.columns
    ]

    if missing_targets:

        raise ValueError(
            "The following target columns were not found: "
            + str(missing_targets)
        )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    feature_columns = [
        c for c in numeric_columns
        if c not in TARGETS
    ]

    # If the dataset contains no separate predictors,
    # use the target variables as input features.
    if len(feature_columns) == 0:
        feature_columns = TARGETS.copy()

    X = df[feature_columns].values.astype(
        np.float32
    )

    y = df[TARGETS].values.astype(
        np.float32
    )

    print("\nInput features:")
    print(feature_columns)

    print("\nPrediction targets:")
    print(TARGETS)

    return X, y, feature_columns


# ============================================================
# 5. Time-ordered train/validation/test split
# ============================================================

def split_data(X, y):

    n = len(X)

    train_end = int(
        TRAIN_RATIO * n
    )

    validation_end = int(
        (TRAIN_RATIO + VALIDATION_RATIO) * n
    )

    X_train = X[:train_end]
    X_validation = X[
        train_end:validation_end
    ]
    X_test = X[validation_end:]

    y_train = y[:train_end]
    y_validation = y[
        train_end:validation_end
    ]
    y_test = y[validation_end:]

    print("\nData split:")
    print("Training:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Testing:", X_test.shape)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ============================================================
# 6. Scaling
# ============================================================

def scale_data(
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test
):

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(
        X_train
    )

    X_validation = x_scaler.transform(
        X_validation
    )

    X_test = x_scaler.transform(
        X_test
    )

    y_train = y_scaler.fit_transform(
        y_train
    )

    y_validation = y_scaler.transform(
        y_validation
    )

    y_test = y_scaler.transform(
        y_test
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        x_scaler,
        y_scaler
    )


# ============================================================
# 7. Create temporal sequences
# ============================================================

def create_sequences(X, y):

    X_sequences = []
    y_sequences = []

    for i in range(
        len(X) - SEQUENCE_LENGTH
    ):

        X_sequences.append(
            X[
                i:i + SEQUENCE_LENGTH
            ]
        )

        y_sequences.append(
            y[
                i + SEQUENCE_LENGTH
            ]
        )

    return (
        np.asarray(
            X_sequences,
            dtype=np.float32
        ),
        np.asarray(
            y_sequences,
            dtype=np.float32
        )
    )


# ============================================================
# 8. GAN augmentation
# ============================================================

def perform_gan_augmentation(
    X_train,
    augmentation_ratio=0.30
):

    number_of_samples = int(
        len(X_train) *
        augmentation_ratio
    )

    print(
        "\nGenerating",
        number_of_samples,
        "synthetic samples using GAN..."
    )

    synthetic_data = generate_synthetic_data(
        X_train,
        number_of_samples
    )

    X_augmented = np.concatenate(
        [
            X_train,
            synthetic_data
        ],
        axis=0
    )

    print(
        "Original training samples:",
        len(X_train)
    )

    print(
        "Augmented training samples:",
        len(X_augmented)
    )

    return X_augmented


# ============================================================
# 9. Evaluation
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    denominator = np.maximum(
        np.abs(y_true),
        1e-8
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred)
            / denominator
        )
    ) * 100

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


# ============================================================
# 10. Main experimental pipeline
# ============================================================

def main():

    print("=" * 70)
    print("GAN-AM-NTM-APSO Environmental Prediction Framework")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_dataset()

    X, y, feature_columns = prepare_data(
        df
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = split_data(X, y)

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        x_scaler,
        y_scaler
    ) = scale_data(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )

    # --------------------------------------------------------
    # GAN augmentation
    # --------------------------------------------------------

    X_train_augmented = \
        perform_gan_augmentation(
            X_train
        )

    # --------------------------------------------------------
    # Temporal sequences
    # --------------------------------------------------------

    X_train_seq, y_train_seq = \
        create_sequences(
            X_train_augmented,
            np.resize(
                y_train,
                (
                    len(X_train_augmented),
                    y_train.shape[1]
                )
            )
        )

    X_validation_seq, y_validation_seq = \
        create_sequences(
            X_validation,
            y_validation
        )

    X_test_seq, y_test_seq = \
        create_sequences(
            X_test,
            y_test
        )

    print(
        "\nSequence shape:",
        X_train_seq.shape
    )

    # --------------------------------------------------------
    # AM-NTM
    # --------------------------------------------------------

    model = build_am_ntm_model(
        input_shape=(
            X_train_seq.shape[1],
            X_train_seq.shape[2]
        ),
        output_dim=len(TARGETS)
    )

    # --------------------------------------------------------
    # APSO
    # --------------------------------------------------------

    print(
        "\nStarting APSO optimization..."
    )

    best_parameters = optimize_model(
        model,
        X_validation_seq,
        y_validation_seq
    )

    print(
        "\nBest APSO parameters:"
    )

    print(best_parameters)

    # --------------------------------------------------------
    # Final training
    # --------------------------------------------------------

    model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(
            X_validation_seq,
            y_validation_seq
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions_scaled = model.predict(
        X_test_seq,
        verbose=0
    )

    predictions = y_scaler.inverse_transform(
        predictions_scaled
    )

    y_test_original = \
        y_scaler.inverse_transform(
            y_test_seq
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    results = {}

    for i, target in enumerate(
        TARGETS
    ):

        result = calculate_metrics(
            y_test_original[:, i],
            predictions[:, i]
        )

        results[target] = result

        print(
            f"\n{target}"
        )

        print(
            f"MAE  : {result['MAE']:.4f}"
        )

        print(
            f"RMSE : {result['RMSE']:.4f}"
        )

        print(
            f"MAPE : {result['MAPE']:.2f}%"
        )

        print(
            f"R2   : {result['R2']:.4f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_rows = []

    for target, values in results.items():

        result_rows.append(
            {
                "Target": target,
                "MAE": values["MAE"],
                "RMSE": values["RMSE"],
                "MAPE (%)": values["MAPE"],
                "R2": values["R2"]
            }
        )

    pd.DataFrame(
        result_rows
    ).to_csv(
        "GAN_AM_NTM_APSO_results.csv",
        index=False
    )

    print(
        "\nResults saved to:"
        " GAN_AM_NTM_APSO_results.csv"
    )


if __name__ == "__main__":
    main()

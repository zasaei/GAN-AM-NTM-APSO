# ============================================================
# GAN.py
# GAN-based augmentation for environmental time-series data
# ============================================================

import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    LeakyReLU,
    Dropout
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# GAN Class
# ============================================================

class EnvironmentalGAN:

    def __init__(
        self,
        n_features,
        latent_dim=100,
        learning_rate=0.0002,
        beta_1=0.5
    ):

        self.n_features = n_features
        self.latent_dim = latent_dim

        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=beta_1
        )

        # ----------------------------------------------------
        # Generator
        # ----------------------------------------------------

        self.generator = self.build_generator()

        # ----------------------------------------------------
        # Discriminator
        # ----------------------------------------------------

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        # ----------------------------------------------------
        # Combined GAN
        # ----------------------------------------------------

        self.discriminator.trainable = False

        noise = Input(
            shape=(self.latent_dim,)
        )

        generated_sample = self.generator(noise)

        validity = self.discriminator(
            generated_sample
        )

        self.combined = tf.keras.Model(
            noise,
            validity
        )

        self.combined.compile(
            loss="binary_crossentropy",
            optimizer=optimizer
        )

    # ========================================================
    # Generator
    # ========================================================

    def build_generator(self):

        model = Sequential(
            name="Environmental_Generator"
        )

        model.add(
            Input(
                shape=(self.latent_dim,)
            )
        )

        model.add(
            Dense(256)
        )

        model.add(
            LeakyReLU(negative_slope=0.2)
        )

        model.add(
            BatchNormalization()
        )

        model.add(
            Dense(512)
        )

        model.add(
            LeakyReLU(negative_slope=0.2)
        )

        model.add(
            BatchNormalization()
        )

        model.add(
            Dense(1024)
        )

        model.add(
            LeakyReLU(negative_slope=0.2)
        )

        model.add(
            BatchNormalization()
        )

        # Output dimension = number of environmental features
        model.add(
            Dense(
                self.n_features,
                activation="tanh"
            )
        )

        return model

    # ========================================================
    # Discriminator
    # ========================================================

    def build_discriminator(self):

        model = Sequential(
            name="Environmental_Discriminator"
        )

        model.add(
            Input(
                shape=(self.n_features,)
            )
        )

        model.add(
            Dense(512)
        )

        model.add(
            LeakyReLU(
                negative_slope=0.2
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            Dense(256)
        )

        model.add(
            LeakyReLU(
                negative_slope=0.2
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            Dense(
                1,
                activation="sigmoid"
            )
        )

        return model

    # ========================================================
    # Training
    # ========================================================

    def train(
        self,
        X_train,
        epochs=500,
        batch_size=64,
        verbose=True
    ):

        X_train = np.asarray(
            X_train,
            dtype=np.float32
        )

        n_samples = X_train.shape[0]

        valid = np.ones(
            (batch_size, 1),
            dtype=np.float32
        )

        fake = np.zeros(
            (batch_size, 1),
            dtype=np.float32
        )

        history = {
            "d_loss": [],
            "d_accuracy": [],
            "g_loss": []
        }

        for epoch in range(epochs):

            # ------------------------------------------------
            # Select real samples
            # ------------------------------------------------

            idx = np.random.randint(
                0,
                n_samples,
                batch_size
            )

            real_samples = X_train[idx]

            # ------------------------------------------------
            # Generate synthetic samples
            # ------------------------------------------------

            noise = np.random.normal(
                0,
                1,
                (
                    batch_size,
                    self.latent_dim
                )
            ).astype(np.float32)

            synthetic_samples = (
                self.generator.predict(
                    noise,
                    verbose=0
                )
            )

            # ------------------------------------------------
            # Train discriminator
            # ------------------------------------------------

            d_loss_real = (
                self.discriminator.train_on_batch(
                    real_samples,
                    valid
                )
            )

            d_loss_fake = (
                self.discriminator.train_on_batch(
                    synthetic_samples,
                    fake
                )
            )

            d_loss = (
                0.5 *
                np.add(
                    d_loss_real,
                    d_loss_fake
                )
            )

            # ------------------------------------------------
            # Train generator
            # ------------------------------------------------

            noise = np.random.normal(
                0,
                1,
                (
                    batch_size,
                    self.latent_dim
                )
            ).astype(np.float32)

            g_loss = (
                self.combined.train_on_batch(
                    noise,
                    valid
                )
            )

            # ------------------------------------------------
            # Store history
            # ------------------------------------------------

            history["d_loss"].append(
                float(d_loss[0])
            )

            history["d_accuracy"].append(
                float(d_loss[1])
            )

            history["g_loss"].append(
                float(g_loss)
            )

            # ------------------------------------------------
            # Display progress
            # ------------------------------------------------

            if verbose and (
                epoch % 50 == 0
                or epoch == epochs - 1
            ):

                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"D loss: {d_loss[0]:.4f} | "
                    f"D acc: {100*d_loss[1]:.2f}% | "
                    f"G loss: {g_loss:.4f}"
                )

        return history

    # ========================================================
    # Generate synthetic data
    # ========================================================

    def generate(
        self,
        n_samples
    ):

        noise = np.random.normal(
            0,
            1,
            (
                n_samples,
                self.latent_dim
            )
        ).astype(np.float32)

        synthetic_data = (
            self.generator.predict(
                noise,
                verbose=0
            )
        )

        return synthetic_data


# ============================================================
# Main function used by GAN-AM-NTM-APSO.py
# ============================================================

def generate_synthetic_data(
    X_train,
    n_samples,
    epochs=500,
    batch_size=64,
    latent_dim=100
):

    X_train = np.asarray(
        X_train,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # GAN expects features in approximately [-1, 1].
    # X_train should therefore be scaled before calling this
    # function.
    # --------------------------------------------------------

    gan = EnvironmentalGAN(
        n_features=X_train.shape[1],
        latent_dim=latent_dim
    )

    print("\nTraining GAN...")
    print(
        f"Real samples : {X_train.shape[0]}"
    )
    print(
        f"Features     : {X_train.shape[1]}"
    )
    print(
        f"Synthetic    : {n_samples}"
    )

    gan.train(
        X_train,
        epochs=epochs,
        batch_size=batch_size
    )

    synthetic_data = gan.generate(
        n_samples
    )

    print(
        "\nSynthetic data generated:",
        synthetic_data.shape
    )

    return synthetic_data


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print(
        "GAN.py is ready."
    )

    print(
        "Use generate_synthetic_data("
        "X_train, n_samples)"
    )

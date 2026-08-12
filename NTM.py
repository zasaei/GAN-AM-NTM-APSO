import tensorflow as tf
from tensorflow.keras import layers


class AdaptiveNTMCell(layers.Layer):
    """Adaptive-memory NTM cell with recurrent controller and external memory."""

    def __init__(
        self,
        input_dim,
        controller_size=128,
        memory_size=64,
        memory_vector_dim=32,
        read_head_num=1,
        write_head_num=1,
        shift_range=1,
        output_dim=None,
        adaptation_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.controller_size = int(controller_size)
        self.memory_size = int(memory_size)
        self.memory_vector_dim = int(memory_vector_dim)
        self.read_head_num = int(read_head_num)
        self.write_head_num = int(write_head_num)
        self.shift_range = int(shift_range)
        self.output_dim = int(output_dim or input_dim)
        self.adaptation_rate = float(adaptation_rate)

        self.controller = layers.GRUCell(self.controller_size)

        n_heads = self.read_head_num + self.write_head_num
        per_head = (
            self.memory_vector_dim + 1 + 1
            + (2 * self.shift_range + 1) + 1
        )
        total = n_heads * per_head + (
            2 * self.write_head_num * self.memory_vector_dim
        )

        self.parameter_projection = layers.Dense(total)
        self.output_projection = layers.Dense(self.output_dim)

        self.memory_gate = layers.Dense(
            self.memory_vector_dim, activation="sigmoid"
        )
        self.memory_update = layers.Dense(
            self.memory_vector_dim, activation="tanh"
        )
        self.adaptation_projection = layers.Dense(
            1, activation="sigmoid"
        )

    def initial_state(self, batch_size, dtype=tf.float32):
        return {
            "controller_state": tf.zeros(
                [batch_size, self.controller_size], dtype=dtype
            ),
            "read_vectors": [
                tf.zeros(
                    [batch_size, self.memory_vector_dim], dtype=dtype
                )
                for _ in range(self.read_head_num)
            ],
            "weights": [
                tf.ones(
                    [batch_size, self.memory_size], dtype=dtype
                ) / self.memory_size
                for _ in range(
                    self.read_head_num + self.write_head_num
                )
            ],
            "memory": tf.random.normal(
                [batch_size, self.memory_size, self.memory_vector_dim],
                stddev=0.05,
                dtype=dtype
            ),
            "usage": tf.ones(
                [batch_size, self.memory_size], dtype=dtype
            ) * 0.5
        }

    def address(
        self, key, beta, gate, shift, gamma,
        memory, previous_weight
    ):
        key_expanded = tf.expand_dims(key, -1)

        cosine = tf.squeeze(
            tf.matmul(memory, key_expanded), -1
        ) / (
            tf.norm(memory, axis=2) + 1e-8
        ) / (
            tf.norm(key, axis=1) + 1e-8
        )

        content_weight = tf.nn.softmax(
            beta[:, None] * cosine, axis=-1
        )

        gated = (
            gate[:, None] * content_weight
            + (1.0 - gate[:, None]) * previous_weight
        )

        shifted = tf.stack(
            [
                tf.roll(gated, delta=delta, axis=1)
                for delta in range(
                    -self.shift_range,
                    self.shift_range + 1
                )
            ],
            axis=-1
        )

        shifted_weight = tf.reduce_sum(
            shifted * shift[:, None, :], axis=-1
        )

        sharpened = tf.pow(
            tf.maximum(shifted_weight, 1e-8),
            gamma[:, None]
        )

        return sharpened / (
            tf.reduce_sum(
                sharpened, axis=1, keepdims=True
            ) + 1e-8
        )

    def call(self, x, state):
        previous_reads = state["read_vectors"]
        controller_state = state["controller_state"]
        memory = state["memory"]
        previous_weights = state["weights"]
        usage = state["usage"]

        controller_input = tf.concat(
            [x] + previous_reads, axis=-1
        )

        controller_output, new_controller_state = (
            self.controller(
                controller_input,
                [controller_state]
            )
        )
        new_controller_state = new_controller_state[0]

        params = self.parameter_projection(
            controller_output
        )

        n_heads = (
            self.read_head_num
            + self.write_head_num
        )

        per_head = (
            self.memory_vector_dim + 1 + 1
            + (2 * self.shift_range + 1) + 1
        )

        heads = tf.split(
            params[:, :n_heads * per_head],
            n_heads,
            axis=-1
        )

        write_params = tf.split(
            params[:, n_heads * per_head:],
            2 * self.write_head_num,
            axis=-1
        )

        weights = []

        for i, head in enumerate(heads):
            key = tf.tanh(
                head[:, :self.memory_vector_dim]
            )

            beta = (
                tf.nn.softplus(
                    head[:, self.memory_vector_dim]
                ) + 1e-3
            )

            gate = tf.sigmoid(
                head[:, self.memory_vector_dim + 1]
            )

            shift_logits = head[
                :,
                self.memory_vector_dim + 2:
                self.memory_vector_dim + 2
                + 2 * self.shift_range + 1
            ]

            shift = tf.nn.softmax(
                shift_logits, axis=-1
            )

            gamma = (
                1.0
                + tf.nn.softplus(head[:, -1])
            )

            weights.append(
                self.address(
                    key,
                    beta,
                    gate,
                    shift,
                    gamma,
                    memory,
                    previous_weights[i]
                )
            )

        read_weights = weights[
            :self.read_head_num
        ]

        write_weights = weights[
            self.read_head_num:
        ]

        reads = [
            tf.reduce_sum(
                w[..., None] * memory,
                axis=1
            )
            for w in read_weights
        ]

        for i, weight in enumerate(
            write_weights
        ):
            erase = tf.sigmoid(
                write_params[2 * i]
            )
            add = tf.tanh(
                write_params[2 * i + 1]
            )

            memory = memory * (
                1.0
                - weight[..., None]
                * erase[:, None, :]
            )

            memory = memory + (
                weight[..., None]
                * add[:, None, :]
            )

        activity = tf.reduce_max(
            tf.stack(write_weights, axis=1),
            axis=1
        )

        new_usage = tf.clip_by_value(
            0.9 * usage + 0.1 * activity,
            0.0,
            1.0
        )

        adaptation_signal = (
            self.adaptation_projection(
                controller_output
            )
        )

        memory_summary = tf.reduce_mean(
            memory, axis=1
        )

        candidate_memory = self.memory_update(
            controller_output
        )

        memory_gate = self.memory_gate(
            tf.concat(
                [memory_summary, controller_output],
                axis=-1
            )
        )

        alpha = (
            self.adaptation_rate
            * adaptation_signal
            * memory_gate
        )

        memory = (
            (1.0 - alpha[:, None, :]) * memory
            + alpha[:, None, :]
            * candidate_memory[:, None, :]
        )

        memory = memory * (
            1.0
            - self.adaptation_rate
            * new_usage
        )[..., None]

        output = self.output_projection(
            tf.concat(
                [controller_output] + reads,
                axis=-1
            )
        )

        return output, {
            "controller_state": new_controller_state,
            "read_vectors": reads,
            "weights": weights,
            "memory": memory,
            "usage": new_usage
        }


class AMNTM(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        controller_size=128,
        memory_size=64,
        memory_vector_dim=32,
        read_head_num=1,
        write_head_num=1,
        shift_range=1,
        output_dim=3,
        adaptation_rate=0.1
    ):
        super().__init__()

        self.input_dim = int(input_dim)

        self.cell = AdaptiveNTMCell(
            input_dim=input_dim,
            controller_size=controller_size,
            memory_size=memory_size,
            memory_vector_dim=memory_vector_dim,
            read_head_num=read_head_num,
            write_head_num=write_head_num,
            shift_range=shift_range,
            output_dim=output_dim,
            adaptation_rate=adaptation_rate
        )

    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, tf.float32)

        if inputs.shape[1] is None:
            raise ValueError(
                "Sequence length must be fixed."
            )

        state = self.cell.initial_state(
            tf.shape(inputs)[0],
            inputs.dtype
        )

        outputs = []

        for t in range(inputs.shape[1]):
            output, state = self.cell(
                inputs[:, t, :],
                state
            )
            outputs.append(output)

        return tf.stack(outputs, axis=1)


def build_am_ntm(
    input_dim,
    output_dim,
    controller_size=128,
    memory_size=64,
    memory_vector_dim=32,
    read_head_num=1,
    write_head_num=1,
    shift_range=1,
    adaptation_rate=0.1,
    learning_rate=1e-3
):
    model = AMNTM(
        input_dim=input_dim,
        controller_size=controller_size,
        memory_size=memory_size,
        memory_vector_dim=memory_vector_dim,
        read_head_num=read_head_num,
        write_head_num=write_head_num,
        shift_range=shift_range,
        output_dim=output_dim,
        adaptation_rate=adaptation_rate
    )

    model(
        tf.zeros(
            (1, 4, input_dim),
            dtype=tf.float32
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="MAE"
            ),
            tf.keras.metrics.RootMeanSquaredError(
                name="RMSE"
            )
        ]
    )

    return model


# Compatibility name for the previous main script.
build_am_ntm_model = build_am_ntm

"""
NTM.py
Adaptive Neural Turing Machine (AM-NTM) for environmental time-series data.

This implementation preserves the main addressing/read/write logic of the
provided legacy NTM implementation while using modern TensorFlow/Keras APIs.
The adaptive-memory component adds a learned memory gate and usage-aware
memory retention so that the external memory can adapt to changing temporal
patterns in environmental data.
"""

import tensorflow as tf
from tensorflow.keras import layers


class AdaptiveNTMCell(layers.Layer):
    """Neural Turing Machine cell with adaptive external memory."""

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        if memory_size < 2:
            raise ValueError("memory_size must be >= 2")
        if memory_vector_dim < 1:
            raise ValueError("memory_vector_dim must be >= 1")
        if read_head_num < 1 or write_head_num < 1:
            raise ValueError("At least one read and one write head are required")
        if not 0.0 < adaptation_rate <= 1.0:
            raise ValueError("adaptation_rate must be in (0, 1]")

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
        per_head = self.memory_vector_dim + 1 + 1 + (2 * self.shift_range + 1) + 1
        total_params = n_heads * per_head + 2 * self.write_head_num * self.memory_vector_dim

        self.parameter_projection = layers.Dense(total_params)
        self.output_projection = layers.Dense(self.output_dim)

        # Adaptive-memory components.
        self.memory_gate = layers.Dense(self.memory_vector_dim, activation="sigmoid")
        self.memory_update = layers.Dense(self.memory_vector_dim, activation="tanh")

        # Usage is used as a soft retention signal. It is stateful per sequence,
        # not a trainable variable.
        self.adaptation_projection = layers.Dense(1, activation="sigmoid")

    def initial_state(self, batch_size, dtype=tf.float32):
        """Create a fresh NTM state for a batch."""
        memory = tf.random.normal(
            [batch_size, self.memory_size, self.memory_vector_dim],
            stddev=0.05,
            dtype=dtype,
        )
        controller_state = tf.zeros([batch_size, self.controller_size], dtype=dtype)
        read_vectors = [
            tf.zeros([batch_size, self.memory_vector_dim], dtype=dtype)
            for _ in range(self.read_head_num)
        ]
        weights = [
            tf.ones([batch_size, self.memory_size], dtype=dtype) / float(self.memory_size)
            for _ in range(self.read_head_num + self.write_head_num)
        ]
        usage = tf.ones([batch_size, self.memory_size], dtype=dtype) * 0.5

        return {
            "controller_state": controller_state,
            "read_vectors": read_vectors,
            "weights": weights,
            "memory": memory,
            "usage": usage,
        }

    def call(self, x, state):
        previous_reads = state["read_vectors"]
        controller_state = state["controller_state"]
        previous_memory = state["memory"]
        previous_weights = state["weights"]
        previous_usage = state["usage"]

        controller_input = tf.concat([x] + previous_reads, axis=-1)
        controller_output, new_controller_state = self.controller(
            controller_input, [controller_state]
        )
        new_controller_state = new_controller_state[0]

        parameters = self.parameter_projection(controller_output)

        n_heads = self.read_head_num + self.write_head_num
        per_head = self.memory_vector_dim + 1 + 1 + (2 * self.shift_range + 1) + 1
        head_params = tf.split(parameters[:, : n_heads * per_head], n_heads, axis=-1)
        erase_add = tf.split(
            parameters[:, n_heads * per_head :],
            2 * self.write_head_num,
            axis=-1,
        )

        weights = []
        addressing_info = []
        for i, p in enumerate(head_params):
            k = tf.tanh(p[:, : self.memory_vector_dim])
            beta = tf.nn.softplus(p[:, self.memory_vector_dim]) + 1e-3
            g = tf.sigmoid(p[:, self.memory_vector_dim + 1])
            shift_logits = p[
                :, self.memory_vector_dim + 2 : self.memory_vector_dim + 2 + 2 * self.shift_range + 1
            ]
            shift = tf.nn.softmax(shift_logits, axis=-1)
            gamma = 1.0 + tf.nn.softplus(p[:, -1])

            w = self.address(k, beta, g, shift, gamma, previous_memory, previous_weights[i])
            weights.append(w)
            addressing_info.append({"k": k, "beta": beta, "g": g, "shift": shift, "gamma": gamma})

        # Read from memory.
        read_weights = weights[: self.read_head_num]
        read_vectors = [
            tf.reduce_sum(w[..., None] * previous_memory, axis=1) for w in read_weights
        ]

        # Write to memory.
        memory = previous_memory
        write_weights = weights[self.read_head_num :]
        for i, w in enumerate(write_weights):
            erase = tf.sigmoid(erase_add[2 * i])
            add = tf.tanh(erase_add[2 * i + 1])
            w_expanded = w[..., None]
            memory = memory * (1.0 - w_expanded * erase[:, None, :])
            memory = memory + w_expanded * add[:, None, :]

        # Adaptive memory: retain important locations and softly refresh less
        # useful locations using the current controller representation.
        write_activity = tf.reduce_max(tf.stack(write_weights, axis=1), axis=1)
        new_usage = tf.clip_by_value(
            0.9 * previous_usage + 0.1 * write_activity, 0.0, 1.0
        )

        adaptation_signal = self.adaptation_projection(controller_output)
        memory_summary = tf.reduce_mean(memory, axis=1)
        candidate = self.memory_update(controller_output)
        gate = self.memory_gate(tf.concat([memory_summary, controller_output], axis=-1))

        # The global adaptive update is deliberately small so it complements,
        # rather than replaces, the NTM write-head mechanism.
        alpha = self.adaptation_rate * adaptation_signal * gate
        memory = (1.0 - alpha[:, None, :]) * memory + alpha[:, None, :] * candidate[:, None, :]

        # Penalize/reduce retention of heavily used locations slightly. This
        # encourages adaptive reuse of external memory rather than saturation.
        retention = 1.0 - self.adaptation_rate * new_usage
        memory = memory * retention[..., None]

        output = self.output_projection(
            tf.concat([controller_output] + read_vectors, axis=-1)
        )

        new_state = {
            "controller_state": new_controller_state,
            "read_vectors": read_vectors,
            "weights": weights,
            "memory": memory,
            "usage": new_usage,
            "addressing": addressing_info,
        }
        return output, new_state

    def address(self, key, beta, gate, shift, gamma, memory, previous_weight):
        """Content + location addressing following the legacy NTM design."""
        key = tf.expand_dims(key, axis=-1)
        inner = tf.matmul(memory, key)
        key_norm = tf.sqrt(tf.reduce_sum(tf.square(key), axis=1, keepdims=True))
        memory_norm = tf.sqrt(tf.reduce_sum(tf.square(memory), axis=2, keepdims=True))
        cosine = tf.squeeze(inner / (memory_norm * key_norm + 1e-8), axis=-1)

        content_logits = beta[:, None] * cosine
        content_weight = tf.nn.softmax(content_logits, axis=-1)

        gate = gate[:, None]
        gated = gate * content_weight + (1.0 - gate) * previous_weight

        # Circular convolution with a small shift kernel.
        shifted = []
        for delta in range(-self.shift_range, self.shift_range + 1):
            shifted.append(tf.roll(gated, shift=delta, axis=1))
        shifted = tf.stack(shifted, axis=-1)
        shifted_weight = tf.reduce_sum(shifted * shift[:, None, :], axis=-1)

        sharpened = tf.pow(tf.maximum(shifted_weight, 1e-8), gamma[:, None])
        return sharpened / (tf.reduce_sum(sharpened, axis=1, keepdims=True) + 1e-8)


class AMNTM(tf.keras.Model):
    """Sequence model wrapping AdaptiveNTMCell.

    Input shape:  (batch, time, features)
    Output shape: (batch, time, output_dim)
    """

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim or input_dim)
        self.cell = AdaptiveNTMCell(
            input_dim=self.input_dim,
            controller_size=controller_size,
            memory_size=memory_size,
            memory_vector_dim=memory_vector_dim,
            read_head_num=read_head_num,
            write_head_num=write_head_num,
            shift_range=shift_range,
            output_dim=self.output_dim,
            adaptation_rate=adaptation_rate,
        )

    def call(self, inputs, training=None, return_state=False):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        batch_size = tf.shape(inputs)[0]
        time_steps = inputs.shape[1]
        if time_steps is None:
            raise ValueError("AMNTM currently requires a fixed sequence length.")

        state = self.cell.initial_state(batch_size, dtype=inputs.dtype)
        outputs = []
        for t in range(time_steps):
            output, state = self.cell(inputs[:, t, :], state)
            outputs.append(output)

        outputs = tf.stack(outputs, axis=1)
        if return_state:
            return outputs, state
        return outputs


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
):
    """Build and compile the AM-NTM model for environmental prediction."""
    model = AMNTM(
        input_dim=input_dim,
        controller_size=controller_size,
        memory_size=memory_size,
        memory_vector_dim=memory_vector_dim,
        read_head_num=read_head_num,
        write_head_num=write_head_num,
        shift_range=shift_range,
        output_dim=output_dim,
        adaptation_rate=adaptation_rate,
    )

    # Build weights before compile by running one dummy sequence.
    model(tf.zeros([1, 1, input_dim], dtype=tf.float32))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="MAE"),
            tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
        ],
    )
    return model


if __name__ == "__main__":
    # Small smoke test; it does not require DataSet.xlsx.
    tf.random.set_seed(42)
    model = build_am_ntm(input_dim=10, output_dim=3)
    x = tf.random.normal([4, 12, 10])
    y = model(x)
    print("AM-NTM output shape:", y.shape)

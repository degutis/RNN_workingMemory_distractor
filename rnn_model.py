import tensorflow as tf
import numpy as np

# Function to generate random sequences of orientations
def generate_sequence(length):
    sequence = []
    for _ in range(length):
        # Choose one of the two orientations randomly
        orientation = np.random.uniform([0, 2*np.pi])
        sequence.append(orientation)
    return np.array(sequence)

# Number of training examples
num_examples = 1000

# Sequence length
sequence_length = 2

# Generate training data
X_train = np.array([generate_sequence(sequence_length) for _ in range(num_examples)])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=180, output_dim=32, input_length=sequence_length),  # Embedding layer for orientations
    tf.keras.layers.LSTM(64, return_sequences=True),  # LSTM layer with 64 units
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create labels
y_train = np.array([1 if orientation1 in sequence else 0 for sequence in X_train])

model.fit(X_train, y_train, epochs=10, batch_size=32)

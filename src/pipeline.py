# -*- coding: utf-8 -*-
"""
Core pipeline for character-level symbolic recurrence biomarker.

Functions for:
  - Text preprocessing and one-hot encoding
  - Recurrence matrix computation
  - Recurrence-to-image conversion
  - Siamese network construction and training
  - Evaluation with stratified k-fold CV
"""

import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Conv2D, MaxPooling2D, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


# ════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ════════════════════════════════════════════════════════════════

def preprocess_text(text):
    """Strip non-alphanumeric characters (except basic punctuation),
    lowercase, return list of characters."""
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)
    text = text.lower()
    return list(text)


def build_vocab(character_lists):
    """Build character vocabulary from a list of character lists.
    Returns (char_to_index, vocab_size)."""
    all_chars = set()
    for chars in character_lists:
        all_chars.update(chars)
    char_list = sorted(list(all_chars))
    char_to_index = {c: i for i, c in enumerate(char_list)}
    return char_to_index, len(char_list)


def characters_to_onehot(characters, char_to_index, vocab_size):
    """Convert a list of characters to one-hot vectors.

    This is the critical encoding step. One-hot ensures every
    non-matching character pair has identical distance (sqrt(2)),
    removing any ordinal relationship between characters. This
    makes the downstream epsilon-thresholded recurrence
    operationally equivalent to exact symbolic equality.

    Without one-hot (i.e., using raw integer codes), the epsilon
    threshold would treat alphabetically adjacent characters as
    partially recurrent, which is linguistically meaningless.
    """
    indices = [char_to_index[c] for c in characters]
    return np.eye(vocab_size)[indices]


# ════════════════════════════════════════════════════════════════
# RECURRENCE COMPUTATION
# ════════════════════════════════════════════════════════════════

def calculate_recurrence_matrix(sequence, epsilon=None):
    """Compute binary recurrence matrix from a sequence of vectors.

    Args:
        sequence: (N, D) array of embedding vectors.
        epsilon: distance threshold. If None, uses 0.1 * std
                 of the pairwise distance matrix.

    Returns:
        (N, N) binary recurrence matrix.
    """
    sequence = np.array(sequence)
    N = sequence.shape[0]
    distance_matrix = np.linalg.norm(
        sequence[:, None, :] - sequence[None, :, :], axis=2
    )
    if epsilon is None:
        epsilon = 0.1 * np.std(distance_matrix)
        if epsilon == 0:
            epsilon = 0.001
    recurrence_matrix = np.zeros((N, N))
    recurrence_matrix[distance_matrix <= epsilon] = 1
    return recurrence_matrix


def recurrence_to_image(recurrence_matrix, target_size=128):
    """Render a recurrence matrix as a grayscale image array.

    Converts the matrix to a (target_size, target_size, 1) float32
    array normalized to [0, 1], without writing to disk.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(recurrence_matrix, cmap='binary', origin='lower')
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)[:, :, :3]
    gray = np.mean(buf, axis=2)
    plt.close(fig)

    img = Image.fromarray(gray.astype(np.uint8), mode='L')
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[:, :, np.newaxis]


def generate_all_images(df, char_to_index, vocab_size, target_size=128):
    """Generate recurrence plot images for all rows in a DataFrame.

    Args:
        df: DataFrame with a 'characters' column (list of chars).
        char_to_index: character-to-index mapping.
        vocab_size: number of unique characters.
        target_size: output image dimension.

    Returns:
        (N, target_size, target_size, 1) float32 array.
    """
    images = []
    for i, (_, row) in enumerate(df.iterrows()):
        onehot = characters_to_onehot(row['characters'], char_to_index, vocab_size)
        rm = calculate_recurrence_matrix(onehot)
        img = recurrence_to_image(rm, target_size=target_size)
        images.append(img)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}")
    return np.array(images)


# ════════════════════════════════════════════════════════════════
# SIAMESE NETWORK
# ════════════════════════════════════════════════════════════════

def create_base_network(input_shape=(128, 128, 1)):
    """CNN base network for the Siamese architecture.
    Output: 128-dimensional embedding."""
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inp)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(inp, x)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred, margin=1.0):
    return K.mean(
        y_true * K.square(y_pred)
        + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
    )


def build_siamese(input_shape=(128, 128, 1), lr=0.001):
    """Build and compile a Siamese network.
    Returns (siamese_model, base_network)."""
    base_net = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_net(input_a)
    processed_b = base_net(input_b)
    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    siamese = Model([input_a, input_b], distance)
    siamese.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=lr))
    return siamese, base_net


def create_pairs(images, labels):
    """Create positive and negative pairs for contrastive learning.

    Positive pairs: same class, adjacent indices.
    Negative pairs: different class, same index position.
    """
    pairs, pair_labels = [], []
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    min_samples = min(len(ci) for ci in class_indices) - 1

    for idx in range(min_samples):
        for c in range(num_classes):
            anchor = class_indices[c][idx]
            positive = class_indices[c][idx + 1]
            pairs.append([images[anchor], images[positive]])
            pair_labels.append(1)

            neg_class = (c + 1) % num_classes
            negative = class_indices[neg_class][idx]
            pairs.append([images[anchor], images[negative]])
            pair_labels.append(0)
    return np.array(pairs), np.array(pair_labels)


def train_siamese_and_embed(X_train, y_train, X_test,
                            input_shape=(128, 128, 1),
                            epochs=20, batch_size=16, seed=42):
    """Train a Siamese network on train data and extract embeddings.

    Args:
        X_train: training images.
        y_train: training labels.
        X_test: test images (embeddings extracted, not used for training).
        input_shape: image dimensions.
        epochs: training epochs.
        batch_size: training batch size.
        seed: random seed for reproducibility.

    Returns:
        (train_embeddings, test_embeddings)
    """
    K.clear_session()
    tf.random.set_seed(seed)

    siamese, base_net = build_siamese(input_shape)
    pairs, pair_labels = create_pairs(X_train, y_train)

    siamese.fit(
        [pairs[:, 0], pairs[:, 1]], pair_labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0
    )

    emb_train = base_net.predict(X_train, verbose=0)
    emb_test = base_net.predict(X_test, verbose=0)
    return emb_train, emb_test

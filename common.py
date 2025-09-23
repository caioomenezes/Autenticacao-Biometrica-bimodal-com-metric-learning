import numpy as np
import tensorflow as tf
from scipy.stats import zscore
from tensorflow.keras import layers, models

# ---------- 1. Definição das partições de sujeitos ----------
# Cada cliente recebe um subconjunto de sujeitos (para simular FedLearning ou clientes distintos)
CLIENT_SUBJECT_PARTITIONS = {
    0: list(range(0, 9)),
    1: list(range(9, 18)),
    2: list(range(18, 27)),
    3: list(range(27, 36)),
    4: list(range(36, 45)),
    5: list(range(45, 54)),
    6: list(range(54, 63)),
    7: list(range(63, 72)),
    8: list(range(72, 81)),
    9: list(range(81, 86)),
}

# ---------- 2. Modelos de embedding (EEG e PPG separadamente) ----------
def get_eeg_embedding_model(embedding_dim=128):
    """
    CNN para EEG. Recebe um segmento (768, 5, 1) e gera um vetor embedding.
    """
    model = models.Sequential(name='eeg_embedding_model')
    model.add(layers.Input(shape=(768, 5, 1)))
    model.add(layers.Conv2D(64, (5, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('selu'))
    model.add(layers.Conv2D(64, (5, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('selu'))
    model.add(layers.MaxPooling2D((3, 1)))
    model.add(layers.Conv2D(64, (3, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('selu'))
    model.add(layers.MaxPooling2D((3, 1)))
    model.add(layers.Conv2D(64, (3, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('selu'))
    model.add(layers.MaxPooling2D((3, 1)))
    model.add(layers.Conv2D(16, (3, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('selu'))
    model.add(layers.MaxPooling2D((3, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='selu'))
    model.add(layers.Dense(35, activation='selu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(embedding_dim, activation=None))
    model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    return model

def get_ppg_embedding_model(embedding_dim=128):
    """
    CNN para PPG. Recebe um segmento (192, 3, 1) e gera um vetor embedding.
    """
    model = models.Sequential(name='ppg_embedding_model')
    model.add(layers.Input(shape=(192, 3, 1)))
    model.add(layers.Conv2D(64, (5, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('selu'))
    model.add(layers.Conv2D(64, (5, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('selu'))
    model.add(layers.MaxPooling2D((3, 1)))
    model.add(layers.Conv2D(64, (3, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('selu'))
    model.add(layers.MaxPooling2D((3, 1)))
    model.add(layers.Conv2D(16, (3, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('selu'))
    model.add(layers.MaxPooling2D((3, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='selu'))
    model.add(layers.Dense(35, activation='selu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(embedding_dim, activation=None))
    model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    return model

def get_bimodal_model(embedding_dim=128, fusion="concatenacao"):
    eeg_net = get_eeg_embedding_model(embedding_dim)
    ppg_net = get_ppg_embedding_model(embedding_dim)

    input_eeg = layers.Input(shape=(768, 5, 1))
    input_ppg = layers.Input(shape=(192, 3, 1))

    emb_eeg = eeg_net(input_eeg)
    emb_ppg = ppg_net(input_ppg)

    if fusion == "concatenacao":
        fusion_emb = layers.Concatenate()([emb_eeg, emb_ppg])
    elif fusion == "soma":
        fusion_emb = layers.Add()([emb_eeg, emb_ppg])
    elif fusion == "media":
        fusion_emb = layers.Average()([emb_eeg, emb_ppg])
    else:
        raise ValueError("Nao deu certo")

    return models.Model(inputs=[input_eeg, input_ppg], outputs=fusion_emb, name="bimodal_model")

class TripletModel(models.Model):

    def __init__(self, embedding_model, margin=0.5):
        super().__init__()
        self.embedding_model = embedding_model
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    def call(self, inputs, training=False):
        return self.embedding_model(inputs, training=training)

    def _compute_triplet_loss(self, anchor, positive, negative):
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)

    def train_step(self, data):
        # unpack (âncora, positivo, negativo) → cada um é um par [EEG, PPG]
        (anchor_eeg, anchor_ppg), (pos_eeg, pos_ppg), (neg_eeg, neg_ppg) = data

        with tf.GradientTape() as tape:
            anchor_emb = self.embedding_model([anchor_eeg, anchor_ppg], training=True)
            pos_emb = self.embedding_model([pos_eeg, pos_ppg], training=True)
            neg_emb = self.embedding_model([neg_eeg, neg_ppg], training=True)
            loss = self._compute_triplet_loss(anchor_emb, pos_emb, neg_emb)

        gradients = tape.gradient(loss, self.embedding_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.embedding_model.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        (anchor_eeg, anchor_ppg), (pos_eeg, pos_ppg), (neg_eeg, neg_ppg) = data
        anchor_emb = self.embedding_model([anchor_eeg, anchor_ppg], training=False)
        pos_emb = self.embedding_model([pos_eeg, pos_ppg], training=False)
        neg_emb = self.embedding_model([neg_eeg, neg_ppg], training=False)
        loss = self._compute_triplet_loss(anchor_emb, pos_emb, neg_emb)
        self.val_loss_tracker.update_state(loss)
        return {"loss": self.val_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

def load_data(dataset_type, data_path_prefix):
    """
    Carrega todos os dados (EEG ou PPG), normalizando e retornando tudo junto.
    """
    if dataset_type not in ['eeg', 'ppg']:
        raise ValueError("dataset_type must be 'eeg' or 'ppg'")

    # Carrega os rótulos dos dados de treino e teste
    y_train = np.load(f'{data_path_prefix}/y_train.npy')
    y_test = np.load(f'{data_path_prefix}/y_test_all.npy')

    # Carrega os dados de treino e teste usando memmap para eficiência de memória
    X_train = np.load(f'{data_path_prefix}/X_train.npy', mmap_mode='r').copy()
    X_test = np.load(f'{data_path_prefix}/X_test_all.npy', mmap_mode='r').copy()

    # Normaliza os dados (z-score)
    if len(X_train) > 0:
        X_train = zscore(X_train, axis=None)
    if len(X_test) > 0:
        X_test = zscore(X_test, axis=None)

    return X_train, y_train, X_test, y_test

def create_triplet_dataset_bimodal(X_eeg, X_ppg, y, batch_size=64):
    """
    Cria um tf.data.Dataset para embeddings bimodais (EEG + PPG),
    gerando trios (âncora, positivo, negativo).
    """
    if len(X_eeg) == 0 or len(X_ppg) == 0:
        return tf.data.Dataset.from_tensor_slices((([], []), ([], []), ([], []))).batch(batch_size)

    # Converte para tensores
    X_eeg_tensor = tf.constant(X_eeg, dtype=tf.float32)
    X_ppg_tensor = tf.constant(X_ppg, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.int64)
    all_indices = tf.range(tf.shape(y_tensor)[0], dtype=tf.int64)

    def generate_triplets(i):
        # Âncora
        eeg_anchor = X_eeg_tensor[i]
        ppg_anchor = X_ppg_tensor[i]
        anchor_label = y_tensor[i]

        # Positivo (mesmo sujeito)
        positive_indices = tf.where((y_tensor == anchor_label) & (all_indices != i))
        if tf.shape(positive_indices)[0] == 0:
            eeg_positive, ppg_positive = eeg_anchor, ppg_anchor
        else:
            pos_idx = tf.random.shuffle(positive_indices)[0][0]
            eeg_positive = X_eeg_tensor[pos_idx]
            ppg_positive = X_ppg_tensor[pos_idx]

        # Negativo (outro sujeito)
        negative_indices = tf.where(y_tensor != anchor_label)
        neg_idx = tf.random.shuffle(negative_indices)[0][0]
        eeg_negative = X_eeg_tensor[neg_idx]
        ppg_negative = X_ppg_tensor[neg_idx]

        return (eeg_anchor, ppg_anchor), (eeg_positive, ppg_positive), (eeg_negative, ppg_negative)

    dataset = tf.data.Dataset.from_tensor_slices(all_indices)
    #dataset = dataset.shuffle(buffer_size=tf.shape(all_indices)[0], reshuffle_each_iteration=True)
    dataset = dataset.shuffle(buffer_size=tf.cast(tf.shape(all_indices)[0], tf.int64), reshuffle_each_iteration=True)
    dataset = dataset.map(generate_triplets, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return dataset

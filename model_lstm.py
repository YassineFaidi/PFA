import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === Chargement des données ===
data = np.load('processed_data.npz')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# === Paramètres ===
VOCAB_SIZE = 10000       # Doit correspondre à celui du tokenizer
EMBEDDING_DIM = 100
MAX_LEN = X_train.shape[1]
BATCH_SIZE = 32
EPOCHS = 10

# === Modèle LSTM simple ===
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Pour classification binaire
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ModelCheckpoint('lstm_model.h5', save_best_only=True)
]

# === Entraînement ===
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === Évaluation ===
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# === Sauvegarde du modèle final ===
model.save("final_lstm_model.h5")
print("✅ Modèle final sauvegardé dans final_lstm_model.h5")

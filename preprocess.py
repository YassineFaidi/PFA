import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Paramètres ===
MAX_VOCAB = 10000       # Nombre max de mots dans le vocabulaire
MAX_LEN = 200           # Longueur max des séquences (tokens)
TEST_SIZE = 0.2         # 20% pour test
MIN_LENGTH = 10         # Payloads trop courts seront ignorés

# === Charger le dataset combiné ===
df = pd.read_csv('data.csv')

# Nettoyage de base
df.drop_duplicates(subset='payload', inplace=True)
df.dropna(subset=['payload'], inplace=True)
df['payload'] = df['payload'].astype(str).str.strip()
df = df[df['payload'].str.len() >= MIN_LENGTH]

# Statistiques de longueur des payloads
df['length'] = df['payload'].apply(lambda x: len(x.split()))
print("Statistiques des longueurs de tokens :")
print(df['length'].describe())

# === Tokenisation ===
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer.fit_on_texts(df['payload'])

sequences = tokenizer.texts_to_sequences(df['payload'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# === Labels ===
labels = np.array(df['label'])

# === Split en train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
)

# === Sauvegarde ===
np.savez_compressed('processed_data.npz', 
    X_train=X_train, y_train=y_train, 
    X_test=X_test, y_test=y_test)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"\n✅ Données vectorisées sauvegardées dans processed_data.npz")
print(f"✅ Tokenizer sauvegardé dans tokenizer.pkl")
print(f"📊 X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"📊 X_test: {X_test.shape}, y_test: {y_test.shape}")

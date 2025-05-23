# modelo.py

import numpy as np
import random
from faker import Faker
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

fake = Faker()

def generar_secuencia_genetica(longitud=8):
    return [random.randint(1, 20) for _ in range(longitud)]

def riesgo_aleatorio():
    return random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]

# Dataset sintético
X = np.array([generar_secuencia_genetica() for _ in range(800)])
y = np.array([riesgo_aleatorio() for _ in range(800)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo RandomForest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Modelo generativo
vocab_size = 21
max_len = 8
embed_dim = 12

sequences = []
next_bases = []
for seq in X:
    for i in range(1, len(seq)):
        sequences.append(seq[:i])
        next_bases.append(seq[i])

sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
next_bases = to_categorical(next_bases, num_classes=vocab_size)

model = Sequential([
    Embedding(vocab_size, embed_dim, input_length=max_len),
    LSTM(64),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(sequences, next_bases, epochs=5, batch_size=32, verbose=0)

base_map = {
    1:"Adenina", 2:"Citosina", 3:"Guanina", 4:"Timina", 5:"Uracilo",
    6:"Elemento X1", 7:"Elemento X2", 8:"Elemento X3", 9:"Elemento X4", 10:"Elemento X5",
    11:"Proteína A", 12:"Proteína B", 13:"Enzima C", 14:"Enzima D",
    15:"Mutación Alta", 16:"Mutación Baja", 17:"Secuencia Críptica",
    18:"Secuencia Activa", 19:"Elemento Regulador", 20:"Elemento Silenciador"
}

def generar_secuencia(model, seed_seq, longitud=max_len):
    result = seed_seq.copy()
    for _ in range(longitud - len(seed_seq)):
        padded = pad_sequences([result], maxlen=max_len, padding='pre')
        pred = model.predict(padded, verbose=0)[0]
        next_base = np.random.choice(range(vocab_size), p=pred)
        if next_base == 0:  # padding
            break
        result.append(next_base)
    return result

def secuencia_a_texto(seq):
    return ", ".join(base_map.get(b, "?") for b in seq)

class BioFakerIA:
    def __init__(self):
        self.fake = Faker()

    def species_name(self):
        return f"{self.fake.last_name()} {self.fake.word().capitalize()}"

    def habitat(self):
        return self.fake.city()

    def description(self):
        return random.choice([
            "Especie endémica con adaptaciones únicas.",
            "Conocida por su rápido crecimiento.",
            "Posee mecanismos defensivos avanzados.",
            "Importante en el ecosistema local.",
            "Presenta una dieta variada y flexible."
        ])

    def generate(self):
        nombre = self.species_name()
        habitat = self.habitat()
        descripcion = self.description()

        seed = generar_secuencia_genetica(3)
        secuencia = generar_secuencia(model, seed)
        texto_gen = secuencia_a_texto(secuencia)

        riesgo = clf.predict([secuencia])[0]
        niveles = {0:"Bajo", 1:"Medio", 2:"Alto"}

        return {
            "especie": nombre,
            "habitat": habitat,
            "descripcion": descripcion,
            "genoma": texto_gen,
            "riesgo": niveles[riesgo]
        }

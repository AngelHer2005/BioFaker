import streamlit as st
import numpy as np
import random
from faker import Faker
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Copia de funciones y clases necesarias del notebook ---

fake = Faker()

def generar_secuencia_genetica(longitud=8, unica=False):
    if unica:
        # Genera una secuencia de bases únicas (sin repeticiones)
        return random.sample(range(1, 21), min(longitud, 20))
    else:
        return [random.randint(1, 20) for _ in range(longitud)]

def riesgo_aleatorio():
    # Ahora las clases están balanceadas
    return random.choice([0, 1, 2])

X = np.array([generar_secuencia_genetica() for _ in range(800)])
y = np.array([riesgo_aleatorio() for _ in range(800)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

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
    1:"Adenina",2:"Citosina",3:"Guanina",4:"Timina",5:"Uracilo",
    6:"Elemento X1",7:"Elemento X2",8:"Elemento X3",9:"Elemento X4",10:"Elemento X5",
    11:"Proteína A",12:"Proteína B",13:"Enzima C",14:"Enzima D",
    15:"Mutación Alta",16:"Mutación Baja",17:"Secuencia Críptica",
    18:"Secuencia Activa",19:"Elemento Regulador",20:"Elemento Silenciador"
}

def generar_secuencia(model, seed_seq, longitud=max_len):
    result = seed_seq.copy()
    for _ in range(longitud - len(seed_seq)):
        padded = pad_sequences([result], maxlen=max_len, padding='pre')
        pred = model.predict(padded, verbose=0)[0]
        # Evita repetir bases ya presentes
        used_bases = set(result)
        # No se puede usar la base 0 (padding)
        available_bases = [i for i in range(1, vocab_size) if i not in used_bases]
        if not available_bases:
            break
        # Crea una nueva distribución de probabilidad solo con las bases disponibles
        filtered_probs = np.array([pred[i] if i in available_bases else 0 for i in range(vocab_size)])
        if filtered_probs.sum() == 0:
            break
        filtered_probs = filtered_probs / filtered_probs.sum()
        next_base = np.random.choice(range(vocab_size), p=filtered_probs)
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
        descs = [
            "Especie endémica con adaptaciones únicas.",
            "Conocida por su rápido crecimiento.",
            "Posee mecanismos defensivos avanzados.",
            "Importante en el ecosistema local.",
            "Presenta una dieta variada y flexible."
        ]
        return random.choice(descs)

    def generate(self):
        nombre = self.species_name()
        habitat = self.habitat()
        descripcion = self.description()

        # Usa semilla sin repeticiones
        seed = generar_secuencia_genetica(3, unica=True)
        secuencia = generar_secuencia(model, seed)
        texto_gen = secuencia_a_texto(secuencia)

        # --- Arreglo: Prepara la secuencia para el clasificador ---
        # Quita ceros (padding) y ajusta longitud a 8
        secuencia_sin_ceros = [b for b in secuencia if b != 0]
        if len(secuencia_sin_ceros) < 8:
            # Rellena con bases aleatorias válidas
            secuencia_sin_ceros += [random.randint(1, 20) for _ in range(8 - len(secuencia_sin_ceros))]
        elif len(secuencia_sin_ceros) > 8:
            secuencia_sin_ceros = secuencia_sin_ceros[:8]
        # Predice el riesgo usando la secuencia ajustada
        riesgo = clf.predict([secuencia_sin_ceros])[0]
        niveles = {0:"Bajo",1:"Medio",2:"Alto"}
        emojis = {0:"🟢", 1:"🟡", 2:"🔴"}

        return {
            "nombre": nombre,
            "habitat": habitat,
            "descripcion": descripcion,
            "genoma": texto_gen,
            "riesgo": niveles[riesgo],
            "emoji_riesgo": emojis[riesgo]
        }

# --- Interfaz Streamlit ---

# Oculta el menú, footer y el "Running" y aplica fondo negro y colores claros
st.markdown("""
    <style>
    body {
        background: #000 !important;
    }
    .stApp {
        background: transparent;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css-1v0mbdj.e115fcil1 {display: none;}
    .stDeployButton {display: none;}
    .st-emotion-cache-1c7y2kd {display: none;}
    /* Botón personalizado */
    button[kind="secondary"], .stButton>button {
        background: linear-gradient(90deg, #339af0 0%, #5c7cfa 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.7rem 1.5rem !important;
        box-shadow: 0 2px 8px 0 rgba(51,154,240,0.15);
        transition: background 0.2s;
    }
    button[kind="secondary"]:hover, .stButton>button:hover {
        background: linear-gradient(90deg, #228be6 0%, #4263eb 100%) !important;
        color: #fff !important;
    }
    /* Tarjeta */
    .bio-card {
        background: #18181b;
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.18);
        padding: 2.5rem 2rem 2rem 2rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        font-size: 1.15rem;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color: #fff; font-size: 3rem;'>🧬 BioFakerIA</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #e0eafc; font-size: 1.3rem;'>"
    "Genera especies ficticias con nombre, hábitat, descripción, genoma sintético y nivel de riesgo usando IA."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("✨ Generar nueva especie", use_container_width=True):
        biofaker = BioFakerIA()
        resultado = biofaker.generate()
        st.markdown(
            f"""
            <div class="bio-card">
                <h2 style="margin-bottom: 0.5rem; color:#91eaff;">🦠 {resultado['nombre']}</h2>
                <p style="margin:0.2rem 0 0.7rem 0;"><b>🏞️ Hábitat:</b> <span style="color:#b2f2ff;">{resultado['habitat']}</span></p>
                <p style="margin:0.2rem 0 0.7rem 0;"><b>📝 Descripción:</b> <span style="color:#fff;">{resultado['descripcion']}</span></p>
                <p style="margin:0.2rem 0 0.7rem 0;"><b>🧬 Genoma sintético:</b> <span style="color:#a5d8ff;">{resultado['genoma']}</span></p>
                <p style="margin:0.2rem 0 0.7rem 0;"><b>⚠️ Nivel de riesgo para humanos:</b> <span style="font-weight:bold;">{resultado['emoji_riesgo']} {resultado['riesgo']}</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='text-align:center; color:#b2b2b2; margin-top:2rem;'>"
            "Haz clic en <b>✨ Generar nueva especie</b> para crear una especie sintética."
            "</div>",
            unsafe_allow_html=True
        )

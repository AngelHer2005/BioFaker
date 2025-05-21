# BioFaker

Este proyecto utiliza inteligencia artificial para generar especies ficticias con secuencias genéticas sintéticas y clasificar su nivel de riesgo para humanos.

## Características

- **Generación de datos sintéticos:** Crea secuencias genéticas aleatorias simulando bases y elementos ficticios.
- **Clasificación de riesgo:** Entrena un modelo RandomForest para predecir el nivel de riesgo de cada secuencia.
- **Modelo generativo LSTM:** Utiliza una red neuronal LSTM para generar nuevas secuencias genéticas.
- **BioFakerIA:** Clase que produce especies ficticias con nombre, hábitat, descripción, genoma y nivel de riesgo.

## Requisitos

- Python 3.7+
- numpy
- scikit-learn
- tensorflow
- faker

Instala las dependencias con:

```bash
pip install numpy scikit-learn tensorflow faker
```

## Uso

Puedes ejecutar el código principal desde el archivo `codigo_ia.py` o explorar el flujo completo y explicaciones en el notebook `codigo_ia.ipynb`.

### Ejemplo rápido

```python
from codigo_ia import BioFakerIA

biofaker = BioFakerIA()
print(biofaker.generate())
```

### Notebook

Abre `codigo_ia.ipynb` en VSCode o Jupyter para ver el paso a paso y explicación detallada.

## Estructura del Proyecto

- `codigo_ia.py`: Código fuente principal.
- `codigo_ia.ipynb`: Notebook explicativo.
- `README.md`: Este archivo.

## Créditos

Desarrollado con fines educativos y demostrativos de IA generativa y clasificación.

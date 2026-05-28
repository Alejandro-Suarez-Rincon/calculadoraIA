# CalculadoraIA

Aplicación de visión por computador que detecta y resuelve operaciones matemáticas simples en tiempo real desde la cámara web, usando redes neuronales entrenadas con TensorFlow.

## Índice

- [Descripción](#descripción)
- [Stack](#stack)
- [Estructura](#estructura)
- [Inicio rápido](#inicio-rápido)
- [Uso](#uso)

## Descripción

El sistema captura video en tiempo real, aplica una corrección de perspectiva sobre una hoja de papel y divide el área visible en tres zonas de reconocimiento: número 1, operador y número 2. Dos modelos de red neuronal convolucional identifican los dígitos y el símbolo matemático, y el resultado se muestra superpuesto en pantalla.

Soporta suma, resta, multiplicación y división.

## Stack

- **Python 3.9**
- **TensorFlow / Keras** — inferencia con modelos CNN preentrenados
- **OpenCV** — captura de video, corrección de perspectiva y renderizado
- **NumPy** — procesamiento de arrays e imágenes
- **Pillow / Matplotlib** — utilidades de preparación del dataset

## Estructura

```text
calculadoraIA/
├── Modelos/
│   ├── numeros.h5          # Modelo CNN para dígitos 0-9
│   ├── signos.h5           # Modelo CNN para operadores + - * /
│   └── DataSetOperaciones/ # Dataset de entrenamiento de operadores
├── Preparacion/
│   ├── blancoNegro.py      # Convierte imágenes del dataset a escala de grises
│   └── Redimencionar.py    # Redimensiona imágenes al tamaño de entrada del modelo
├── Red/
│   ├── camaraNumeros.py    # Programa principal — cámara en tiempo real
│   ├── cargaNumeros.py     # Entrenamiento del modelo de dígitos
│   └── cargaSignos.py      # Entrenamiento del modelo de operadores
├── run.sh                  # Script de arranque
└── requirements.txt
```

## Inicio rápido

El repositorio incluye los modelos preentrenados. Solo necesitas ejecutar:

```bash
bash run.sh
```

El script crea automáticamente el entorno virtual si no existe, instala las dependencias y arranca el programa. Si falla al iniciar, reintenta la instalación de dependencias antes de rendirse.

### Instalación manual (opcional)

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
python Red/camaraNumeros.py
```

## Uso

1. Al abrir el programa verás la imagen de tu cámara.
2. Haz clic en las **4 esquinas de una hoja de papel** para definir el área de detección.
3. Coloca sobre el papel un número, un operador y otro número, cada uno dentro de su zona resaltada.
4. El resultado aparece en pantalla cuando el modelo supera el umbral de confianza.

| Tecla | Acción |
|---|---|
| `R` | Reiniciar la selección de esquinas |
| `Q` | Salir |

### Reentrenar los modelos

Si quieres regenerar los modelos desde cero:

```bash
# Modelo de dígitos (MNIST)
python Red/cargaNumeros.py

# Modelo de operadores
python Red/cargaSignos.py
```

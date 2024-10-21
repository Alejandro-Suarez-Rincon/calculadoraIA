<h1 style="text-align: center; font-size: 36px;">
    Calculadora IA
</h1>

# Index
- [Index](#index)
- [Descripción](#descripción)
- [Tecnologias](#tecnologias)
- [Estructura](#estructura)
  - [Modelos](#modelos)
  - [Preparacion](#preparacion)
  - [Red](#red)
- [Guía de inicio rápido](#guía-de-inicio-rápido)

# Descripción
Este repositorio contiene un proyecto que demuestra el uso de redes neuronales
para la detección y resolución de ecuaciones matemáticas simples
(suma, resta, multiplicación y división) a través de la cámara del computador.
El sistema reconoce las ecuaciones capturadas y muestra la solución directamente
en pantalla.

# Tecnologias
- Python
- Tensor-Flow

# Estructura
El proyecto se organiza en tres directorios principales: `Modelos`, `Preparacion` y `Red`.
Cada uno desempeña un rol fundamental en el correcto funcionamiento del programa, el cual
se ejecuta de manera sencilla con Python, sin necesidad de Kubernetes u otras herramientas
complejas.

```text
/calculadoraIA
├── Modelos/
|   ├── DataSetOperaciones
|   ├── numeros.h5
|   └── signos.h5
├── Preparacion
│   ├── blancoNegro.py
│   └── Redimencionar.py
└── Red
    ├── camaraNumeros.py
    ├── cargaNumeros.py
    └── cargaSignos.py
```

## Modelos
El directorio `Modelos` contiene los modelos preentrenados para el reconocimiento de números
y signos matemáticos, almacenados en archivos `.h5`.

## Preparacion
En el directorio `Preparacion` se encuentran los scripts necesarios para el procesamiento
y preparación del conjunto de datos `DataSet` de las operaciones básicas. Estos scripts
incluyen la conversión de imágenes a blanco y negro y la redimensión de las mismas.

## Red
El directorio `Red` se encarga del entrenamiento de los modelos. Los scripts aquí permiten
entrenar las redes neuronales y optimizarlas para que `camaraNumeros.py` pueda utilizarlas
de manera eficiente. Esto facilita entrenar las redes múltiples veces y elegir el mejor
modelo resultante.

# Guía de inicio rápido
Este repositorio ya incluye modelos preentrenados, por lo que solo necesitas seguir los
pasos a continuación para ejecutar el proyecto.

1. Asegurarse de tener las librerias listas para usarse
   - cv2
   - numpy
   - tensorflow
   - pathlib
   - matplotlib
   - PIL
  
  Se puede instalar con el siguiente comando

  ```bash
  pip install opencv-python numpy tensorflow pathlib matplotlib pillow
  ```

1. Dentro de `./Red/camaraNumeros.py` ejecutar como un programa normal de python
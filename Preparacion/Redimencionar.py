import os
from PIL import Image
from pathlib import Path

# Obtener la ruta base del archivo actual
base_dir = Path(__file__).resolve().parent

# Ruta del directorio base que contiene las subcarpetas con las imágenes
directorio_base = base_dir / '../Modelos/DataSetOperaciones/'

# Tamaño deseado para las imágenes redimensionadas
nuevo_ancho = 28
nuevo_alto = 28

# Obtener la lista de subcarpetas en el directorio base
subcarpetas = [f for f in os.listdir(directorio_base) if os.path.isdir(directorio_base / f)]

# Iterar sobre cada subcarpeta
for subcarpeta in subcarpetas:
    ruta_subcarpeta = directorio_base / subcarpeta

    # Recorre todas las imágenes en la subcarpeta
    for nombre_archivo in os.listdir(ruta_subcarpeta):
        # Verifica si el archivo es una imagen (extensión común)
        if nombre_archivo.endswith((".jpg", ".jpeg", ".png")):
            ruta_imagen = ruta_subcarpeta / nombre_archivo

            # Abre la imagen utilizando Pillow
            imagen = Image.open(ruta_imagen)

            # Redimensiona la imagen
            imagen_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto))

            # Guarda la imagen redimensionada (sobrescribe la imagen original)
            imagen_redimensionada.save(ruta_imagen)

    print(f"Imágenes redimensionadas en la carpeta: {subcarpeta}")

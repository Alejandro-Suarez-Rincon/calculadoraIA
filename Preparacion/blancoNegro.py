import os
import cv2
from pathlib import Path

# Obtener la ruta base del archivo actual
base_dir = Path(__file__).resolve().parent

# Ruta de la carpeta que contiene las subcarpetas (0, 1, 2, 3)
carpeta_base = base_dir / '../Modelos/DataSetOperaciones/'

# Obtener la lista de subcarpetas (0, 1, 2, 3)
subcarpetas = [f for f in os.listdir(carpeta_base) if os.path.isdir(carpeta_base / f)]

# Iterar sobre cada subcarpeta
for subcarpeta in subcarpetas:
    ruta_subcarpeta = carpeta_base / subcarpeta

    # Obtener la lista de archivos en la subcarpeta
    archivos = os.listdir(ruta_subcarpeta)

    # Iterar sobre cada archivo en la subcarpeta
    for archivo in archivos:
        # Comprobar si es un archivo de imagen
        if archivo.endswith((".jpg", ".jpeg", ".png")):
            # Ruta completa de la imagen
            ruta_imagen = ruta_subcarpeta / archivo

            # Leer la imagen en escala de grises
            imagen = cv2.imread(str(ruta_imagen), 0)

            # Guardar la imagen en blanco y negro
            nombre_salida = archivo.split(".")[0] + ".png"  # Nombre de salida
            ruta_salida = ruta_subcarpeta / nombre_salida
            cv2.imwrite(str(ruta_salida), imagen)

    print(f"Procesamiento completado en la carpeta: {subcarpeta}")

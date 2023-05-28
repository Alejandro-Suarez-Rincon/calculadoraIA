import cv2
import os

# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = "C:\Alejandro\Programacion\Python\calculadoraIA\Modelos\DataSetOperaciones\Suma"

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta_imagenes)

# Iterar sobre cada archivo en la carpeta
for archivo in archivos:
    # Comprobar si es un archivo de imagen
    if archivo.endswith((".jpg", ".jpeg", ".png")):
        # Ruta completa de la imagen
        ruta_imagen = os.path.join(carpeta_imagenes, archivo)

        # Leer la imagen en escala de grises
        imagen = cv2.imread(ruta_imagen, 0)

        # Guardar la imagen en blanco y negro
        nombre_salida = archivo.split(".")[0] + ".jpg"  # Nombre de salida
        ruta_salida = os.path.join(carpeta_imagenes, nombre_salida)
        cv2.imwrite(ruta_salida, imagen)

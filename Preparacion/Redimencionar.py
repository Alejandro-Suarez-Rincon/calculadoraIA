import os

from PIL import Image

# Ruta del directorio que contiene las imágenes
directorio_imagenes = 'C:/Alejandro/Programacion/Python/calculadoraIA/Modelos/DataSetOperaciones/2'

# Tamaño deseado para las imágenes redimensionadas
nuevo_ancho = 28
nuevo_alto = 28

# Recorre todas las imágenes en el directorio
for nombre_archivo in os.listdir(directorio_imagenes):
    # Verifica si el archivo es una imagen (extensión común)
    if nombre_archivo.endswith(".jpg") or nombre_archivo.endswith(".jpeg") or nombre_archivo.endswith(".png"):
        ruta_imagen = os.path.join(directorio_imagenes, nombre_archivo)

        # Abre la imagen utilizando Pillow
        imagen = Image.open(ruta_imagen)

        # Redimensiona la imagen
        imagen_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto))

        # Guarda la imagen redimensionada (sobreescribe la imagen original)
        imagen_redimensionada.save(ruta_imagen)

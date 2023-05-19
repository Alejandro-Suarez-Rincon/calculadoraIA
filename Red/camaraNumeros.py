import cv2
import numpy as np
from tensorflow import keras

# Cargar el modelo pre-entrenado
print("Cargando el modelo")
model = keras.models.load_model('C:/Modelos/numeros.h5')

print("Modelo cargado")


# Función para preprocesar la imagen de entrada
def preprocess(img):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar un filtro gaussiano para suavizar la imagen
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Aplicar un umbral adaptativo para binarizar la imagen
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Redimensionar la imagen al tamaño requerido por el modelo (28x28)
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    # Convertir la imagen a un arreglo numpy de tipo float32 y normalizar los valores a un rango de 0 a 1
    img_array = np.array(resized, dtype=np.float32) / 255.0
    # Cambiar la forma del arreglo para que tenga un tamaño de (1, 28, 28, 1)
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

def preprocess2(img):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar un filtro gaussiano para suavizar la imagen
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Aplicar un umbral adaptativo para binarizar la imagen
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Redimensionar la imagen al tamaño requerido por el modelo (28x28)
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    # Convertir la imagen a un arreglo numpy de tipo float32 y normalizar los valores a un rango de 0 a 1
    img_array = np.array(resized, dtype=np.float32) / 255.0
    # Cambiar la forma del arreglo para que tenga un tamaño de (1, 28, 28, 1)
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

#image = cv2.imread('C:/Users/aleja/Desktop/operaciones.png')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #frame = image.copy()
    # Dibujar un rectángulo para definir la región de interés (ROI)
    cv2.rectangle(frame, (100, 200), (200, 300), (255, 0, 0), 2)
    cv2.rectangle(frame, (250, 200), (350, 300), (0, 0, 255), 2)
    cv2.rectangle(frame, (400, 200), (500, 300), (0, 255, 0), 2)
    # Obtener la imagen de la ROI
    roi1 = frame[200:300, 100:200]  # Esquina superiror, esquina inferior
    roi2 = frame[200:300, 250:350]  # Esquina superiror, esquina inferior
    roi3 = frame[200:300, 400:500]  # Esquina superiror, esquina inferior
    # Preprocesar la imagen de la ROI
    img_array = preprocess(roi1)
    img_array2 = preprocess(roi3)
    # Realizar la predicción utilizando el modelo pre-entrenado
    prediction = model.predict(img_array)
    prediction2 = model.predict(img_array2)
    # Obtener el número predicho
    number = np.argmax(prediction)
    number2 = np.argmax(prediction2)
    # Escribir el número en la pantalla
    cv2.putText(frame, str(number), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(frame, str(number2), (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    respuesta = int(number)  # "*" # (number)
    cv2.imshow("Calculadora con IA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

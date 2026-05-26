import cv2
import numpy as np
from tensorflow import keras
from pathlib import Path

base_dir = Path(__file__).resolve().parent

model_path  = base_dir / '../Modelos/numeros.h5'
model2_path = base_dir / '../Modelos/signos.h5'

ROI_W = 150
ROI_H = 150
CONFIDENCE_THRESHOLD = 0.7

# Mapeo de índice de clase a símbolo del operador (0=-, 1=+, 2=/, 3=*)
OPERATORS = {0: '-', 1: '+', 2: '/', 3: '*'}

try:
    print("Cargando modelos...")
    model  = keras.models.load_model(model_path)
    model2 = keras.models.load_model(model2_path)
    print("Modelos cargados")
except Exception as e:
    raise RuntimeError(f"Error al cargar los modelos: {e}")


def preprocess(img):
    """Convierte una imagen ROI al formato de entrada requerido por los modelos (1, 28, 28, 1)."""
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh  = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    img_array = np.array(resized, dtype=np.float32) / 255.0
    return img_array.reshape((1, 28, 28, 1))


def put_text_centered(frame, text, cx, y, font_scale=1.5, color=(0, 165, 255), thickness=2):
    """Dibuja texto centrado horizontalmente en la coordenada cx."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(frame, text, (cx - tw // 2, y), font, font_scale, color, thickness)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Centrar los tres ROI horizontal y verticalmente en la ventana
x_start = (frame_w - 3 * ROI_W) // 2
y_start = (frame_h - ROI_H) // 2
y_end   = y_start + ROI_H

ROI_FIRST    = (x_start,             x_start + ROI_W,     y_start, y_end)
ROI_OPERATOR = (x_start + ROI_W,     x_start + 2 * ROI_W, y_start, y_end)
ROI_SECOND   = (x_start + 2 * ROI_W, x_start + 3 * ROI_W, y_start, y_end)

LABEL_Y  = y_start - 15   # Etiquetas encima de los cuadros
RESULT_Y = y_end + 55     # Resultado debajo de los cuadros

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    frame = cv2.flip(frame, 1)  # Corregir efecto espejo

    x1s, x1e, y1s, y1e = ROI_FIRST
    x2s, x2e, y2s, y2e = ROI_OPERATOR
    x3s, x3e, y3s, y3e = ROI_SECOND

    cv2.rectangle(frame, (x1s, y1s), (x1e, y1e), (255, 0, 0), 2)
    cv2.rectangle(frame, (x2s, y2s), (x2e, y2e), (0, 0, 255), 2)
    cv2.rectangle(frame, (x3s, y3s), (x3e, y3e), (0, 255, 0), 2)

    roi_first    = frame[y1s:y1e, x1s:x1e]
    roi_operator = frame[y2s:y2e, x2s:x2e]
    roi_second   = frame[y3s:y3e, x3s:x3e]

    first_array    = preprocess(roi_first)
    operator_array = preprocess(roi_operator)
    second_array   = preprocess(roi_second)

    prediction_first    = model.predict(first_array,    verbose=0)
    prediction_operator = model2.predict(operator_array, verbose=0)
    prediction_second   = model.predict(second_array,   verbose=0)

    first_number   = np.argmax(prediction_first)
    operator_index = np.argmax(prediction_operator)
    second_number  = np.argmax(prediction_second)

    confidence_first    = np.max(prediction_first)
    confidence_operator = np.max(prediction_operator)
    confidence_second   = np.max(prediction_second)

    cx1 = (x1s + x1e) // 2
    cx2 = (x2s + x2e) // 2
    cx3 = (x3s + x3e) // 2
    cx_all = (x1s + x3e) // 2  # Centro de los tres cuadros juntos

    put_text_centered(frame, str(first_number),  cx1, LABEL_Y)
    put_text_centered(frame, str(second_number), cx3, LABEL_Y)

    all_confident = (
        confidence_first    >= CONFIDENCE_THRESHOLD and
        confidence_operator >= CONFIDENCE_THRESHOLD and
        confidence_second   >= CONFIDENCE_THRESHOLD
    )

    if all_confident:
        operator_symbol = OPERATORS.get(operator_index, '?')
        put_text_centered(frame, operator_symbol, cx2, LABEL_Y)

        if operator_index == 0:
            result = first_number - second_number
        elif operator_index == 1:
            result = first_number + second_number
        elif operator_index == 2:
            result = round(first_number / second_number, 2) if second_number != 0 else "Err"
        elif operator_index == 3:
            result = first_number * second_number
        else:
            result = "?"

        put_text_centered(frame, f"= {result}", cx_all, RESULT_Y, font_scale=2.0, color=(0, 255, 0))

    cv2.imshow("Calculadora con IA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

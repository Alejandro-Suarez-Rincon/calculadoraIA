import cv2
import numpy as np
from tensorflow import keras
from pathlib import Path

base_dir = Path(__file__).resolve().parent
model_path  = base_dir / '../Modelos/numeros.h5'
model2_path = base_dir / '../Modelos/signos.h5'

WARPED_W = 600
WARPED_H = 400
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


def order_corners(pts):
    """Ordena 4 puntos como: top-left, top-right, bottom-right, bottom-left."""
    pts_sorted = sorted(pts, key=lambda p: p[1])
    top    = sorted(pts_sorted[:2], key=lambda p: p[0])
    bottom = sorted(pts_sorted[2:], key=lambda p: p[0])
    return [top[0], top[1], bottom[1], bottom[0]]


corners = []
M = None  # Matriz de transformación de perspectiva

DST_PTS = np.float32([
    [0,           0          ],
    [WARPED_W - 1, 0          ],
    [WARPED_W - 1, WARPED_H - 1],
    [0,           WARPED_H - 1],
])


def on_mouse_click(event, x, y, flags, param):
    global corners, M
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x, y))
        if len(corners) == 4:
            src_pts = np.float32(order_corners(corners))
            M = cv2.getPerspectiveTransform(src_pts, DST_PTS)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

cv2.namedWindow("Calculadora con IA")
cv2.setMouseCallback("Calculadora con IA", on_mouse_click)

# ROI centrados sobre la imagen corregida
x_start = (WARPED_W - 3 * ROI_W) // 2
y_start = (WARPED_H - ROI_H) // 2
y_end   = y_start + ROI_H

ROI_FIRST    = (x_start,             x_start + ROI_W,     y_start, y_end)
ROI_OPERATOR = (x_start + ROI_W,     x_start + 2 * ROI_W, y_start, y_end)
ROI_SECOND   = (x_start + 2 * ROI_W, x_start + 3 * ROI_W, y_start, y_end)

LABEL_Y  = y_start - 15
RESULT_Y = y_end + 55

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    frame = cv2.flip(frame, 1)

    if M is None:
        display = frame.copy()

        for i, pt in enumerate(corners):
            cv2.circle(display, pt, 8, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(corners) >= 2:
            for i in range(len(corners) - 1):
                cv2.line(display, corners[i], corners[i + 1], (0, 255, 0), 2)

        msg = f"Haz clic en las 4 esquinas del papel  ({len(corners)}/4)   |   R = reiniciar"
        cv2.putText(display, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("Calculadora con IA", display)

    else:
        warped = cv2.warpPerspective(frame, M, (WARPED_W, WARPED_H))
        warped = cv2.flip(warped, 0)  # Invertir en Y para que los números escritos queden derechos

        x1s, x1e, y1s, y1e = ROI_FIRST
        x2s, x2e, y2s, y2e = ROI_OPERATOR
        x3s, x3e, y3s, y3e = ROI_SECOND

        cv2.rectangle(warped, (x1s, y1s), (x1e, y1e), (255, 0, 0), 2)
        cv2.rectangle(warped, (x2s, y2s), (x2e, y2e), (0, 0, 255), 2)
        cv2.rectangle(warped, (x3s, y3s), (x3e, y3e), (0, 255, 0), 2)

        roi_first    = warped[y1s:y1e, x1s:x1e]
        roi_operator = warped[y2s:y2e, x2s:x2e]
        roi_second   = warped[y3s:y3e, x3s:x3e]

        first_array    = preprocess(roi_first)
        operator_array = preprocess(roi_operator)
        second_array   = preprocess(roi_second)

        prediction_first    = model.predict(first_array,     verbose=0)
        prediction_operator = model2.predict(operator_array, verbose=0)
        prediction_second   = model.predict(second_array,    verbose=0)

        first_number   = np.argmax(prediction_first)
        operator_index = np.argmax(prediction_operator)
        second_number  = np.argmax(prediction_second)

        confidence_first    = np.max(prediction_first)
        confidence_operator = np.max(prediction_operator)
        confidence_second   = np.max(prediction_second)

        cx1   = (x1s + x1e) // 2
        cx2   = (x2s + x2e) // 2
        cx3   = (x3s + x3e) // 2
        cx_all = (x1s + x3e) // 2

        put_text_centered(warped, str(first_number),  cx1, LABEL_Y)
        put_text_centered(warped, str(second_number), cx3, LABEL_Y)

        all_confident = (
            confidence_first    >= CONFIDENCE_THRESHOLD and
            confidence_operator >= CONFIDENCE_THRESHOLD and
            confidence_second   >= CONFIDENCE_THRESHOLD
        )

        if all_confident:
            operator_symbol = OPERATORS.get(operator_index, '?')
            put_text_centered(warped, operator_symbol, cx2, LABEL_Y)

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

            put_text_centered(warped, f"= {result}", cx_all, RESULT_Y,
                              font_scale=2.0, color=(0, 255, 0))

        cv2.putText(warped, "R = reiniciar esquinas  |  Q = salir",
                    (10, WARPED_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        cv2.imshow("Calculadora con IA", warped)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        corners = []
        M = None

cap.release()
cv2.destroyAllWindows()

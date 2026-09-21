"""
live_detect.py
----------------
Captura en tiempo real una ventana especifica de Windows (por ejemplo, la
ventana donde corre Clash Royale) y muestra las detecciones de tu modelo
YOLO entrenado en una ventana superpuesta.

REQUISITOS (instalar en tu venv de Ultralytics):
    pip install ultralytics mss opencv-python pygetwindow numpy

USO:
    1. Ajusta las variables de configuracion mas abajo (WINDOW_TITLE, MODEL_PATH).
    2. Abre el juego / la aplicacion en una ventana visible (no minimizada).
    3. Ejecuta:  python live_detect.py
    4. Pulsa 'q' en la ventana de video para cerrar.
"""

import time

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

try:
    import pygetwindow as gw
except ImportError:
    gw = None


# ============ CONFIGURACION ============

# Texto (o parte del texto) del titulo de la ventana a capturar.
# Ejemplos: "Clash Royale", "BlueStacks", "LDPlayer"
WINDOW_TITLE = "Clash Royale"

# Ruta a tu modelo entrenado
MODEL_PATH = r"C:\DragonAI\runs\detect\train-3\weights\best.pt"

# Umbral de confianza minimo para mostrar una deteccion
CONFIDENCE_THRESHOLD = 0.5

# Si no se encuentra la ventana por titulo, se captura esta region fija
# (izquierda, arriba, ancho, alto) como respaldo. Ajusta a tu pantalla.
FALLBACK_REGION = {"left": 0, "top": 0, "width": 1280, "height": 720}

# Cada cuantos frames se refresca la busqueda de la posicion de la ventana
# (por si la mueves o cambias de tamano mientras juegas)
REFRESH_WINDOW_EVERY_N_FRAMES = 30

# Titulo de la ventana de vista previa que abre este script. Debe ser
# distinto de WINDOW_TITLE para que el script no se capture a si mismo.
PREVIEW_WINDOW_TITLE = "YOLO Live Preview (pulsa 'q' para salir)"

# =========================================


def find_window_region(title_substring, exclude_substring=None):
    """Busca una ventana cuyo titulo contenga el texto dado y devuelve
    su region como dict compatible con mss. Devuelve None si no la encuentra.
    Excluye cualquier ventana cuyo titulo contenga exclude_substring, para
    evitar que el script capture su propia ventana de vista previa."""
    if gw is None:
        return None

    matches = [
        w for w in gw.getAllWindows()
        if title_substring.lower() in w.title.lower()
        and (not exclude_substring or exclude_substring.lower() not in w.title.lower())
    ]
    if not matches:
        return None

    win = matches[0]
    if win.isMinimized:
        return None

    # Asegura valores no negativos y con area valida
    left, top = max(win.left, 0), max(win.top, 0)
    width, height = win.width, win.height
    if width <= 0 or height <= 0:
        return None

    return {"left": left, "top": top, "width": width, "height": height}


def main():
    print(f"Cargando modelo desde: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"Modelo cargado con {len(class_names)} clases.")

    sct = mss()
    region = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE) or FALLBACK_REGION
    print(f"Capturando region inicial: {region}")

    frame_count = 0
    prev_time = time.time()

    while True:
        # Refresca la posicion de la ventana periodicamente por si se movio
        if frame_count % REFRESH_WINDOW_EVERY_N_FRAMES == 0:
            new_region = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE)
            if new_region:
                region = new_region

        # Captura el frame de la region actual
        raw = np.array(sct.grab(region))
        # mss devuelve BGRA, convertimos a BGR para OpenCV/YOLO
        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        # Inferencia
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        # Dibuja las detecciones sobre el frame
        annotated = results[0].plot()

        # Calcula y muestra FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow(PREVIEW_WINDOW_TITLE, annotated)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

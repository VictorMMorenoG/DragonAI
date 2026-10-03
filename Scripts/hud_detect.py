"""
hud_detect.py
--------------
Calibra y lee el HUD de Clash Royale (barra de elixir + slots de cartas
en mano) a partir de la captura en vivo de la ventana del juego.

No usa YOLO ni GPU: la barra de elixir se lee por color de pixeles y
los slots de cartas se recortan (el reconocimiento por plantillas se
anade en un paso posterior, una vez tengas capturas limpias de cada carta).

REQUISITOS (mismo venv que live_detect.py):
    pip install mss opencv-python pygetwindow numpy

USO:
    1. Calibrar una vez (marca con el raton las regiones sobre tu ventana):
           python hud_detect.py --calibrate

    2. Una vez calibrado, ejecutar en modo deteccion en vivo:
           python hud_detect.py

    Pulsa 'q' en cualquier ventana de video para salir.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from mss import mss

try:
    import pygetwindow as gw
except ImportError:
    gw = None


# ============ CONFIGURACION ============

WINDOW_TITLE = "Clash Royale"
CONFIG_PATH = "hud_config.json"
TEMPLATES_DIR = "templates"

# Ventanas de vista previa: nombres que NO deben coincidir con WINDOW_TITLE,
# para que el script no intente capturarse a si mismo.
PREVIEW_WINDOW_TITLE = "HUD Detect - Calibracion/Deteccion (pulsa 'q' para salir)"

FALLBACK_REGION = {"left": 0, "top": 0, "width": 1280, "height": 720}

# Color aproximado (BGR) de la barra de elixir llena. AJUSTA esto tras ver
# tu propia captura: haz una captura, abre con un editor de imagenes y
# comprueba el color exacto (o usa el modo --calibrate, que te lo indicara).
ELIXIR_COLOR_BGR = np.array([200, 50, 200])
ELIXIR_TOLERANCIA = 45

REFRESH_WINDOW_EVERY_N_FRAMES = 30

# =========================================


def find_window_region(title_substring, exclude_substring=None):
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
    left, top = max(win.left, 0), max(win.top, 0)
    width, height = win.width, win.height
    if width <= 0 or height <= 0:
        return None
    return {"left": left, "top": top, "width": width, "height": height}


def capturar_frame(sct, region):
    raw = np.array(sct.grab(region))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


# ---------- MODO CALIBRACION ----------

class CalibradorRegiones:
    """Permite marcar rectangulos con click+arrastre sobre una imagen fija
    y devuelve las coordenadas en proporciones relativas (0.0-1.0)."""

    def __init__(self, frame):
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.regiones = {}

    def _click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)

    def pedir_region(self, nombre_region, instrucciones):
        self.start_point = None
        self.end_point = None
        ventana = PREVIEW_WINDOW_TITLE
        cv2.namedWindow(ventana)
        cv2.setMouseCallback(ventana, self._click_callback)

        while True:
            vista = self.frame.copy()
            cv2.putText(
                vista, instrucciones, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
            # Dibuja regiones ya calibradas, en verde
            for nombre, (x1p, y1p, x2p, y2p) in self.regiones.items():
                p1 = (int(x1p * self.w), int(y1p * self.h))
                p2 = (int(x2p * self.w), int(y2p * self.h))
                cv2.rectangle(vista, p1, p2, (0, 200, 0), 2)
                cv2.putText(vista, nombre, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            # Dibuja el rectangulo que se esta arrastrando ahora, en azul
            if self.start_point and self.end_point:
                cv2.rectangle(vista, self.start_point, self.end_point, (255, 150, 0), 2)

            cv2.imshow(ventana, vista)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                raise KeyboardInterrupt("Calibracion cancelada por el usuario")

            # Enter confirma el rectangulo actual
            if key == 13 and self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                if x2 - x1 < 3 or y2 - y1 < 3:
                    print("Rectangulo demasiado pequeno, intenta de nuevo.")
                    continue
                region_rel = (x1 / self.w, y1 / self.h, x2 / self.w, y2 / self.h)
                self.regiones[nombre_region] = region_rel
                return region_rel


def modo_calibrar():
    sct = mss()
    region = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE) or FALLBACK_REGION
    print(f"Capturando ventana: {region}")
    print("Se tomara una captura fija para calibrar. Asegurate de que la")
    print("partida/menu este visible con el HUD (elixir y cartas) en pantalla.")
    time.sleep(1.5)

    frame = capturar_frame(sct, region)
    calibrador = CalibradorRegiones(frame)

    tareas = [
        ("elixir_barra", "Marca la BARRA DE ELIXIR completa (vacia->llena) y pulsa Enter"),
        ("carta_slot_1", "Marca el SLOT DE CARTA 1 (primero a la izquierda) y pulsa Enter"),
        ("carta_slot_2", "Marca el SLOT DE CARTA 2 y pulsa Enter"),
        ("carta_slot_3", "Marca el SLOT DE CARTA 3 y pulsa Enter"),
        ("carta_slot_4", "Marca el SLOT DE CARTA 4 (ultimo) y pulsa Enter"),
    ]

    try:
        for nombre, instrucciones in tareas:
            calibrador.pedir_region(nombre, instrucciones)
            print(f"  -> {nombre}: {calibrador.regiones[nombre]}")
    except KeyboardInterrupt as e:
        print(str(e))
        cv2.destroyAllWindows()
        return

    cv2.destroyAllWindows()

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(calibrador.regiones, f, indent=2)

    print(f"\nCalibracion guardada en: {os.path.abspath(CONFIG_PATH)}")
    print("Ya puedes ejecutar 'python hud_detect.py' (sin --calibrate) para probar la deteccion en vivo.")


# ---------- MODO DETECCION ----------

def cargar_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"No se encontro '{CONFIG_PATH}'. Ejecuta primero: python hud_detect.py --calibrate"
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_plantillas():
    """Carga todas las imagenes de la carpeta templates/ como un dict
    {nombre_carta: imagen}. Devuelve un dict vacio si no existe la carpeta
    o no hay plantillas todavia."""
    plantillas = {}
    if not os.path.isdir(TEMPLATES_DIR):
        return plantillas
    for archivo in os.listdir(TEMPLATES_DIR):
        if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
            nombre = os.path.splitext(archivo)[0]
            img = cv2.imread(os.path.join(TEMPLATES_DIR, archivo))
            if img is not None:
                plantillas[nombre] = img
    return plantillas


def identificar_carta(slot_img, plantillas, umbral=0.35):
    """Compara el recorte de un slot contra todas las plantillas cargadas
    y devuelve (nombre_mas_parecido, score) o (None, 0) si no hay match
    suficientemente bueno o el recorte es invalido."""
    if slot_img is None or slot_img.shape[0] < 5 or slot_img.shape[1] < 5 or not plantillas:
        return None, 0.0

    mejor_nombre, mejor_score = None, -1.0
    for nombre, plantilla in plantillas.items():
        # Redimensiona la plantilla al tamano del slot capturado para comparar
        plantilla_ajustada = cv2.resize(plantilla, (slot_img.shape[1], slot_img.shape[0]))
        resultado = cv2.matchTemplate(slot_img, plantilla_ajustada, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(resultado)
        if score > mejor_score:
            mejor_score = score
            mejor_nombre = nombre

    if mejor_score < umbral:
        return None, mejor_score
    return mejor_nombre, mejor_score


def region_relativa_a_pixeles(region_rel, width, height):
    x1p, y1p, x2p, y2p = region_rel
    return int(x1p * width), int(y1p * height), int(x2p * width), int(y2p * height)


def leer_elixir(frame, region_rel_barra, width, height):
    x1, y1, x2, y2 = region_relativa_a_pixeles(region_rel_barra, width, height)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    franja = frame[y1:y2, x1:x2]
    mascara = np.all(np.abs(franja.astype(int) - ELIXIR_COLOR_BGR) < ELIXIR_TOLERANCIA, axis=-1)
    columnas_con_relleno = np.any(mascara, axis=0)
    porcentaje = columnas_con_relleno.sum() / max(franja.shape[1], 1)
    return round(porcentaje * 10, 1)


def modo_deteccion():
    config = cargar_config()
    plantillas = cargar_plantillas()
    if plantillas:
        print(f"Cargadas {len(plantillas)} plantillas de cartas: {', '.join(plantillas.keys())}")
    else:
        print("Aviso: no se encontraron plantillas en 'templates/'. Solo se mostraran los recortes,")
        print("sin reconocer el nombre de la carta. Ejecuta capture_templates.py para crearlas.")

    sct = mss()
    region = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE) or FALLBACK_REGION
    print(f"Capturando region inicial: {region}")

    frame_count = 0
    while True:
        if frame_count % REFRESH_WINDOW_EVERY_N_FRAMES == 0:
            nueva = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE)
            if nueva:
                region = nueva

        frame = capturar_frame(sct, region)
        h, w = frame.shape[:2]

        elixir = leer_elixir(frame, tuple(config["elixir_barra"]), w, h)

        vista = frame.copy()
        cv2.putText(
            vista, f"Elixir: {elixir}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )

        # Dibuja y recorta los slots de cartas, e identifica cada una
        # contra las plantillas cargadas (si las hay)
        recortes = []
        nombres_detectados = []
        for i in range(1, 5):
            key = f"carta_slot_{i}"
            if key not in config:
                continue
            x1, y1, x2, y2 = region_relativa_a_pixeles(tuple(config[key]), w, h)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            cv2.rectangle(vista, (x1, y1), (x2, y2), (255, 150, 0), 2)
            if x2 > x1 and y2 > y1:
                recorte = frame[y1:y2, x1:x2]
                recortes.append(recorte)
                nombre, score = identificar_carta(recorte, plantillas)
                etiqueta = f"{nombre} ({score:.2f})" if nombre else "?"
                nombres_detectados.append(etiqueta)
                cv2.putText(
                    vista, etiqueta, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if nombre else (0, 0, 255), 2,
                )

        cv2.imshow(PREVIEW_WINDOW_TITLE, vista)

        # Muestra los recortes de cartas en una ventana aparte, uno al lado del otro
        if recortes:
            alto_comun = 100
            redimensionados = [
                cv2.resize(r, (int(r.shape[1] * alto_comun / r.shape[0]), alto_comun))
                for r in recortes if r.shape[0] > 0 and r.shape[1] > 0
            ]
            if redimensionados:
                tira = np.hstack(redimensionados)
                cv2.imshow("Cartas detectadas (recorte)", tira)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibracion y deteccion de HUD de Clash Royale")
    parser.add_argument("--calibrate", action="store_true", help="Ejecuta el modo de calibracion interactiva")
    args = parser.parse_args()

    if args.calibrate:
        modo_calibrar()
    else:
        modo_deteccion()
"""
capture_templates.py
---------------------
Captura plantillas limpias de tus cartas directamente desde el menu de
mazo/coleccion del juego (donde se ven todas quietas, bien iluminadas y
sin superposiciones de cooldown), marcandolas con el raton una por una.

Las plantillas se guardan en 'templates/' para que hud_detect.py las
reconozca despues en los slots de cartas en mano, con matchTemplate.

REQUISITOS: mismos que hud_detect.py (mss, opencv-python, pygetwindow, numpy)

USO:
    1. Abre el juego en la pantalla de mazo/coleccion, donde se vean
       claramente las cartas que quieres capturar.
    2. Ejecuta:  python capture_templates.py
    3. Se abrira una ventana con una captura fija de esa pantalla.
    4. Para cada carta: haz click y arrastra un rectangulo ajustado al
       icono de la carta (sin texto ni bordes de fuera), pulsa Enter,
       y escribe su nombre en la consola.
    5. Repite con todas las cartas que veas en pantalla. Si tienes mas
       cartas que capturar y no caben todas en una captura, cierra con
       'q', desplazate en el juego a la siguiente pantalla de cartas,
       y vuelve a ejecutar el script (no borra las plantillas anteriores).
    6. Pulsa 'q' en la ventana de imagen cuando termines.
"""

import os

import cv2
import numpy as np
from mss import mss

try:
    import pygetwindow as gw
except ImportError:
    gw = None


WINDOW_TITLE = "Clash Royale"
TEMPLATES_DIR = "templates"
PREVIEW_WINDOW_TITLE = "Capturar Plantillas de Cartas (Enter=guardar, q=salir)"
FALLBACK_REGION = {"left": 0, "top": 0, "width": 1280, "height": 720}


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


class SelectorCartas:
    """Permite marcar rectangulos con click+arrastre sobre una imagen fija
    y guardarlos como plantillas nombradas."""

    def __init__(self, frame):
        self.frame_original = frame
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.guardadas = []  # lista de (nombre, (x1,y1,x2,y2)) ya confirmadas, para dibujarlas en verde

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

    def ejecutar(self):
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        cv2.namedWindow(PREVIEW_WINDOW_TITLE)
        cv2.setMouseCallback(PREVIEW_WINDOW_TITLE, self._click_callback)

        print("Marca cada carta con click+arrastre y pulsa Enter para guardarla.")
        print("Pulsa 'q' sobre la ventana de imagen cuando termines.\n")

        while True:
            vista = self.frame_original.copy()
            cv2.putText(
                vista, "Enter=guardar carta seleccionada | q=salir", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

            for nombre, (x1, y1, x2, y2) in self.guardadas:
                cv2.rectangle(vista, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(vista, nombre, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            if self.start_point and self.end_point:
                cv2.rectangle(vista, self.start_point, self.end_point, (255, 150, 0), 2)

            cv2.imshow(PREVIEW_WINDOW_TITLE, vista)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break

            if key == 13 and self.start_point and self.end_point:  # Enter
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                if x2 - x1 < 5 or y2 - y1 < 5:
                    print("Rectangulo demasiado pequeno, intenta de nuevo.")
                    continue

                recorte = self.frame_original[y1:y2, x1:x2]
                nombre = input("Nombre de esta carta (ej: giant, knight, musketeer): ").strip().lower().replace(" ", "_")
                if not nombre:
                    print("Nombre vacio, se descarta esta captura.\n")
                    continue

                ruta = os.path.join(TEMPLATES_DIR, f"{nombre}.png")
                cv2.imwrite(ruta, recorte)
                print(f"Guardada plantilla: {ruta}\n")

                self.guardadas.append((nombre, (x1, y1, x2, y2)))
                self.start_point = None
                self.end_point = None

        cv2.destroyAllWindows()


def main():
    sct = mss()
    region = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE) or FALLBACK_REGION
    print(f"Capturando ventana: {region}")
    print("Asegurate de estar en la pantalla de MAZO/COLECCION con las cartas visibles.\n")

    raw = np.array(sct.grab(region))
    frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    selector = SelectorCartas(frame)
    selector.ejecutar()

    print("Listo. Revisa la carpeta 'templates/' con tus plantillas guardadas.")


if __name__ == "__main__":
    main()
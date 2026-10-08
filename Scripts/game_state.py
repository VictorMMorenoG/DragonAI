"""
game_state.py
--------------
Une todo el pipeline de percepcion en un solo bucle: captura UN frame por
iteracion, lo pasa tanto al modelo YOLO (tropas/torres) como al lector de
HUD (elixir + cartas en mano), y construye un objeto GameState coherente
con todo el estado de la partida en ese instante.

Este es el modulo base sobre el que despues se construye la logica de
decision (reglas, arbol de comportamiento, etc.), que leera GameState en
cada frame para decidir que jugar.

REQUISITOS:
    pip install ultralytics mss opencv-python pygetwindow numpy

REQUIERE HABER HECHO ANTES:
    - Calibracion del HUD:      python hud_detect.py --calibrate
    - Plantillas de cartas:     python capture_templates.py

USO:
    python game_state.py

    Pulsa 'q' en la ventana de video para salir.
"""

import json
import os
import time
from dataclasses import dataclass, field

import pyautogui
import math
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

from accion import jugar_carta

try:
    import pygetwindow as gw
except ImportError:
    gw = None



# ============ CONFIGURACION ============

WINDOW_TITLE = "Clash Royale"
MODEL_PATH = r"C:\DragonAI\runs\detect\train-9\weights\best.pt"
HUD_CONFIG_PATH = "hud_config.json"
TEMPLATES_DIR = "templates"

CONFIDENCE_THRESHOLD_YOLO = 0.5
UMBRAL_MATCH_CARTAS = 0.3
ELIXIR_COLOR_BGR = np.array([200, 50, 200])
ELIXIR_TOLERANCIA = 45
MITAD_CAMPO_Y = 0.43
UMBRAL_CERCA_TORRE_Y = 0.5

COOLDOWN_SEGUNDOS = 2


ZONAS_Y = {
    "detras_torre": 0.75,
    "frente_torre": 0.62,
    "puente": 0.48,
}

FALLBACK_REGION = {"left": 0, "top": 0, "width": 1280, "height": 720}
PREVIEW_WINDOW_TITLE = "GameState - Vista unificada (pulsa 'q' para salir)"
REFRESH_WINDOW_EVERY_N_FRAMES = 30

TORRES = {
    "king_tower_player", "king_tower_enemy",
    "princess_tower_player", "princess_tower_enemy",
}

SWARMS_ENEMY = {"minions_enemy", "archers_enemy", "goblins_enemy", "spear_goblins_enemy", "bat_enemy", "bomber_enemy", "electro_spirit_enemy", "fire_spirit_enemy"}

TROPAS_OBJETIVO_FIREBALL = {"archers_enemy", "musketeer_enemy", "barbarian_enemy", "squeleton_dragon_enemy", "wizard_enemy", "witch_enemy"}

AIR_COUNTERS_ENEMY = {"archers_enemy", "musketeer_enemy", "wizard_enemy", "baby_dragon_enemy", "flying_machine_enemy", "witch_enemy"}

TANQUES_ENEMY = {"giant_enemy", "battle_ram_enemy", "hog_rider_enemy"}

MINITANQUES_ENEMY = {"knight_enemy", "minipekka_enemy", "goblin_brawler_enemy", "valkyrie_enemy", "megaminion_enemy", "pekka_enemy"}

ESTRUCUTAS_ENEMY = {"bomb_tower_enemy", "canon_enemy", "goblin_cage_enemy", "goblin_hut_enemy", "inferno_tower_enemy", "tombstone_enemy"}

TANQUES = {"giant_player"}

MINITANQUES = {"knight_player", "minipekka_player"}

AIR_COUNTERS = {"archers_player", "musketeer_player"}

HECHIZOS = {"arrows", "fireball"}

RADIO_AGRUPACION_HECHIZO = {
    "arrows": 0.15,
    "fireball": 0.10,
}
CARTAS_TANQUE = {"giant"}
CARTAS_MINITANQUE = {"knight", "minipekka", "minions"}
CARTAS_AIR_COUNTER = {"musketeer", "archers"}

OFFSET_DETRAS_TROPA = 0.08

TEMPLATE_REMATCH_PATH = r"C:\DragonAI\templates\jugar_de_nuevo.png"
UMBRAL_MATCH_REMATCH = 0.75
REVISAR_REMATCH_CADA_N_FRAMES = 15  # no hace falta comprobarlo en cada frame

RADIO_CERCANIA_ALIADA_HECHIZO = 0.1

# =========================================


# ---------- ESTRUCTURA DE DATOS ----------

@dataclass
class Tropa:
    tipo: str
    x_norm: float  # 0.0-1.0 relativo al ancho de la ventana capturada
    y_norm: float  # 0.0-1.0 relativo al alto de la ventana capturada
    confianza: float


@dataclass
class GameState:
    timestamp: float
    elixir: float
    mano: list  # lista de 4 nombres de carta (o None si no se reconoce)
    tropas: list = field(default_factory=list)  # lista de objetos Tropa

    @property
    def tropas_enemigas(self):
        """Tropas del enemigo en el campo, sin contar torres."""
        return [t for t in self.tropas if t.tipo.endswith("_enemy") and t.tipo not in TORRES]

    @property
    def tropas_aliadas(self):
        """Tus propias tropas en el campo, sin contar torres."""
        return [t for t in self.tropas if t.tipo.endswith("_player") and t.tipo not in TORRES]

    def resumen(self):
        mano_txt = ", ".join(c if c else "?" for c in self.mano)
        tropas_txt = ", ".join(f"{t.tipo}({t.x_norm:.2f},{t.y_norm:.2f})" for t in self.tropas) or "ninguna"
        return f"Elixir={self.elixir} | Mano=[{mano_txt}] | Tropas: {tropas_txt}"


# ---------- CAPTURA DE VENTANA ----------

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

def buscar_boton_rematch(frame, plantilla, umbral=UMBRAL_MATCH_REMATCH):
    """Busca la plantilla del boton 'Jugar de Nuevo' en todo el frame.
    Devuelve (encontrado, centro_x_pixel, centro_y_pixel, score)."""
    resultado = cv2.matchTemplate(frame, plantilla, cv2.TM_CCOEFF_NORMED)
    _, score, _, max_loc = cv2.minMaxLoc(resultado)

    if score < umbral:
        return False, None, None, score

    h, w = plantilla.shape[:2]
    centro_x = max_loc[0] + w // 2
    centro_y = max_loc[1] + h // 2
    return True, centro_x, centro_y, score

# ---------- HUD: ELIXIR Y CARTAS ----------

def cargar_hud_config():
    if not os.path.exists(HUD_CONFIG_PATH):
        raise FileNotFoundError(
            f"No se encontro '{HUD_CONFIG_PATH}'. Ejecuta primero: python hud_detect.py --calibrate"
        )
    with open(HUD_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_plantillas():
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


def identificar_carta(slot_img, plantillas, umbral=UMBRAL_MATCH_CARTAS):
    if slot_img is None or slot_img.shape[0] < 5 or slot_img.shape[1] < 5 or not plantillas:
        return None, 0.0
    mejor_nombre, mejor_score = None, -1.0
    for nombre, plantilla in plantillas.items():
        plantilla_ajustada = cv2.resize(plantilla, (slot_img.shape[1], slot_img.shape[0]))
        resultado = cv2.matchTemplate(slot_img, plantilla_ajustada, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(resultado)
        if score > mejor_score:
            mejor_score = score
            mejor_nombre = nombre
    if mejor_score < umbral:
        return None, mejor_score
    return mejor_nombre, mejor_score

class TorreTracker:
    """Detecta cuando una princess tower enemiga ha sido destruida: deja
    de verse de forma sostenida tras haber sido confirmada como presente
    durante un tiempo (evita falsos positivos por un frame fallido o una
    tropa tapandola momentaneamente)."""

    FRAMES_CONFIRMAR_VISTA = 10
    FRAMES_AUSENTE_PARA_DESTRUIDA = 30

    def __init__(self):
        self.vista_consecutiva = {"izquierda": 0, "derecha": 0}
        self.ausente_consecutiva = {"izquierda": 0, "derecha": 0}
        self.confirmada = {"izquierda": False, "derecha": False}
        self.destruida = {"izquierda": False, "derecha": False}

    def actualizar(self, tropas):
        vista_este_frame = {"izquierda": False, "derecha": False}

        for t in tropas:
            if t.tipo == "princess_tower_enemy":
                lado = "izquierda" if t.x_norm < 0.5 else "derecha"
                vista_este_frame[lado] = True

        for lado in ("izquierda", "derecha"):
            if self.destruida[lado]:
                continue

            if vista_este_frame[lado]:
                self.vista_consecutiva[lado] += 1
                self.ausente_consecutiva[lado] = 0
                if self.vista_consecutiva[lado] >= self.FRAMES_CONFIRMAR_VISTA:
                    self.confirmada[lado] = True
            else:
                self.vista_consecutiva[lado] = 0
                if self.confirmada[lado]:
                    self.ausente_consecutiva[lado] += 1
                    if self.ausente_consecutiva[lado] >= self.FRAMES_AUSENTE_PARA_DESTRUIDA:
                        self.destruida[lado] = True
                        print(f"[torres] Princess tower enemiga {lado.upper()} destruida")


# ---------- CONSTRUCCION DEL GAMESTATE ----------

def construir_game_state(frame, modelo_yolo, hud_config, plantillas):
    h, w = frame.shape[:2]

    # --- Elixir ---
    elixir = leer_elixir(frame, tuple(hud_config["elixir_barra"]), w, h)

    # --- Cartas en mano ---
    mano = []
    for i in range(1, 5):
        key = f"carta_slot_{i}"
        if key not in hud_config:
            mano.append(None)
            continue
        x1, y1, x2, y2 = region_relativa_a_pixeles(tuple(hud_config[key]), w, h)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        recorte = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
        nombre, _ = identificar_carta(recorte, plantillas)
        mano.append(nombre)

    # --- Tropas/torres (YOLO) ---
    resultados = modelo_yolo.predict(source=frame, conf=CONFIDENCE_THRESHOLD_YOLO, verbose=False)
    tropas = []
    for box in resultados[0].boxes:
        cls_id = int(box.cls[0])
        tipo = modelo_yolo.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # centro de la caja
        tropas.append(Tropa(tipo=tipo, x_norm=round(cx / w, 3), y_norm=round(cy / h, 3), confianza=round(conf, 2)))

    return GameState(timestamp=time.time(), elixir=elixir, mano=mano, tropas=tropas), resultados

# ---------- LOGICA DE CARTAS BASICA ----------

def hay_aliada_cerca(centro, estado: GameState, radio=RADIO_CERCANIA_ALIADA_HECHIZO) -> bool:
    """Comprueba si alguna tropa aliada esta dentro de 'radio' del centro dado."""
    x_centro, y_centro = centro
    return any(
        math.hypot(t.x_norm - x_centro, t.y_norm - y_centro) < radio
        for t in estado.tropas_aliadas
    )


def encontrar_grupo_mas_numeroso(tropas, radio):
    """Busca el grupo de tropas mas numeroso dentro de un radio dado, y
    devuelve (cantidad, posicion_centro) del grupo. Si no hay tropas,
    devuelve (0, None)."""
    if not tropas:
        return 0, None

    mejor_cantidad = 0
    mejor_centro = None

    for t1 in tropas:
        grupo = [t1]
        for t2 in tropas:
            if t2 is t1:
                continue
            distancia = math.hypot(t1.x_norm - t2.x_norm, t1.y_norm - t2.y_norm)
            if distancia < radio:
                grupo.append(t2)

        if len(grupo) > mejor_cantidad:
            mejor_cantidad = len(grupo)
            # Centro del grupo = promedio de posiciones de sus tropas
            centro_x = sum(t.x_norm for t in grupo) / len(grupo)
            centro_y = sum(t.y_norm for t in grupo) / len(grupo)
            mejor_centro = (round(centro_x, 3), round(centro_y, 3))

    return mejor_cantidad, mejor_centro

def val_giant(estado: GameState):
    valor = 0.1

    if not estado.tropas_enemigas:
        valor += 0.1

    if estado.elixir > 8:
        valor += 0.1

    if estado.tropas_aliadas:
        valor += 0.1

    # +0.2 si hay mosqueteras o arqueras aliadas en mi lado de la arena
    aircounters_aliados = any(
        t.tipo in AIR_COUNTERS and t.y_norm > 0.5
        for t in estado.tropas_aliadas
    )
    if aircounters_aliados:
        valor += 0.2
    
    return round(min(valor, 1.0), 2)

def val_knight(estado: GameState):
    valor = 0.1

    if estado.tropas_aliadas:
        valor += 0.2

    # +0.2 si hay un tanque enemigo en mi lado de la arena
    tanques_en_mi_lado = any(
        t.tipo in (TANQUES_ENEMY | MINITANQUES_ENEMY) and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if tanques_en_mi_lado:
        valor += 0.2
    
    # +0.3 si hay un antiaereo enemigo en mi lado de la arena
    aircounters_en_mi_lado = any(
        t.tipo in (AIR_COUNTERS) and t.y_norm > MITAD_CAMPO_Y-0.1
        for t in estado.tropas_enemigas
    )
    if aircounters_en_mi_lado:
        valor += 0.3

    return round(min(valor, 1.0), 2)

def val_musketeer(estado: GameState):
    valor = 0.2

    # +0.2 si hay un tanque enemigo en mi lado de la arena
    tanques_en_mi_lado = any(
        t.tipo in (TANQUES_ENEMY | MINITANQUES_ENEMY) and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if tanques_en_mi_lado:
        valor += 0.2

    # +0.2 si hay swarms enemigos en mi lado de la arena
    swarms_en_mi_lado = any(
        t.tipo in SWARMS_ENEMY and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if swarms_en_mi_lado:
        valor += 0.2

    # +0.2 si hay un tanque aliado en mi lado de la arena
    tanques_aliados = any(
        t.tipo in (TANQUES | MINITANQUES)
        for t in estado.tropas_aliadas
    )
    if tanques_aliados:
        valor += 0.2
    
    return round(min(valor, 1.0), 2)

def val_arrows(estado: GameState):
    valor = 0.0

    tropas_enemigas = [t for t in estado.tropas_enemigas if t.y_norm > 0.3]

    # +0.4 si hay mas de 2 tropas enemigas agrupadas (cerca unas de otras) y hay cartas que mueren a flechas
    cantidad, _ = encontrar_grupo_mas_numeroso(tropas_enemigas, radio=0.15)
    if cantidad > 2 and any(t.tipo in SWARMS_ENEMY for t in tropas_enemigas):
        valor += 0.25
    if cantidad > 4 and any(t.tipo in SWARMS_ENEMY for t in tropas_enemigas):
        valor += 0.25
    if cantidad > 6:
        valor += 0.4
    return round(min(valor, 1.0), 2)

def val_fireball(estado: GameState):
    valor = 0.0

    tropas_enemigas = [t for t in estado.tropas_enemigas if t.y_norm > 0.3]

    # +0.4 si hay mas de 2 tropas enemigas agrupadas (cerca unas de otras) y hay tropas enemigas que mueren a fireball
    cantidad, _ = encontrar_grupo_mas_numeroso(tropas_enemigas, radio=0.1)
    if cantidad > 1 and any(t.tipo in TROPAS_OBJETIVO_FIREBALL for t in tropas_enemigas):
        valor += 0.25
    if cantidad > 2 and any(t.tipo in TROPAS_OBJETIVO_FIREBALL for t in tropas_enemigas):
        valor += 0.25

    return round(min(valor, 1.0), 2)

def val_minipekka(estado: GameState):
    valor = 0.1

    if estado.elixir > 8:
        valor += 0.1

    # +0.5 si hay un giant enemigo en mi lado de la arena
    giant_en_mi_lado = any(
        t.tipo in TANQUES_ENEMY and t.y_norm > 0.4
        for t in estado.tropas_enemigas
    )
    if giant_en_mi_lado:
        valor += 0.5

    # +0.3 si hay mosqueteras o arqueras enemigos en mi lado de la arena
    aircounters_en_mi_lado = any(
        t.tipo in AIR_COUNTERS_ENEMY and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if aircounters_en_mi_lado:
        valor += 0.3

    # +0.3 si hay minitanques enemigos en mi lado de la arena
    minitanques_en_mi_lado = any(
        t.tipo in MINITANQUES_ENEMY and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if minitanques_en_mi_lado:
        valor += 0.3

    # +0.2 si hay un tanque aliado en mi lado de la arena
    tanques_aliados = any(
        t.tipo in (TANQUES)
        for t in estado.tropas_aliadas
    )
    if tanques_aliados:
        valor += 0.2

    # -0.4 si hay swarms enemigos en mi lado de la arena
    swarms_en_mi_lado = any(
        t.tipo in SWARMS_ENEMY and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if swarms_en_mi_lado:
        valor -= 0.4
    
    return round(min(valor, 1.0), 2)

def val_minions(estado: GameState):
    valor = 0.1

    if estado.elixir > 8:
        valor += 0.1

    # +0.3 si hay un tanque enemigo en mi lado de la arena
    tanques_en_mi_lado = any(
        t.tipo in (TANQUES_ENEMY | MINITANQUES_ENEMY) and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if tanques_en_mi_lado:
        valor += 0.3

    if any(t.tipo in AIR_COUNTERS_ENEMY for t in estado.tropas_enemigas):
        valor -= 0.3
    
    # +0.2 si hay un tanque aliado en mi lado de la arena
    tanques_aliados = any(
        t.tipo in (TANQUES | MINITANQUES) and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_aliadas
    )
    if tanques_aliados:
        valor += 0.2

    
    return round(min(valor, 1.0), 2)

def val_archers(estado: GameState):
    valor = 0.2

    # +0.2 si hay un tanque enemigo en mi lado de la arena
    tanques_en_mi_lado = any(
        t.tipo in (TANQUES_ENEMY | MINITANQUES_ENEMY) and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if tanques_en_mi_lado:
        valor += 0.2

    # +0.2 si hay swarms enemigos en mi lado de la arena
    swarms_en_mi_lado = any(
        t.tipo in SWARMS_ENEMY and t.y_norm > 0.5
        for t in estado.tropas_enemigas
    )
    if swarms_en_mi_lado:
        valor += 0.2

    # +0.2 si hay un tanque aliado en mi lado de la arena
    tanques_aliados = any(
        t.tipo in (TANQUES | MINITANQUES)
        for t in estado.tropas_aliadas
    )
    if tanques_aliados:
        valor += 0.2

    return round(min(valor, 1.0), 2)

def valor_esperar(estado: GameState):
    valor = 0.3

    if estado.elixir > 8:
        valor = 0.1
    elif estado.elixir > 9:
        valor = 0.0

    return valor

def calcular_posicion_objetivo_hechizo(nombre_carta: str, estado: GameState):
    """Devuelve la posicion (x_norm, y_norm) donde apuntar un hechizo:
    el centro del grupo enemigo mas numeroso. Si hay una tropa aliada
    cerca del grupo, se tira sin offset (para no arriesgarse a fallar el
    area y dañarla); si no hay aliados cerca, se aplica un pequeño offset
    en y para compensar el movimiento de las tropas enemigas hacia mi
    campo. Si no hay enemigos, apunta a una torre enemiga (fallback)."""
    tropas_enemigas = estado.tropas_enemigas
    radio = RADIO_AGRUPACION_HECHIZO.get(nombre_carta, 0.12)

    _, centro = encontrar_grupo_mas_numeroso(tropas_enemigas, radio)
    if centro is not None:
        x_centro, y_centro = centro

        if hay_aliada_cerca(centro, estado):
            y_ajustado = y_centro  # sin offset
        else:
            y_ajustado = y_centro + 0.07

        y_ajustado = max(0.0, min(1.0, y_ajustado))
        return (x_centro, y_ajustado)

    # Fallback: si no hay tropas enemigas visibles, apunta a una torre enemiga
    torres_enemigas = [t for t in estado.tropas if t.tipo in TORRES and t.tipo.endswith("_enemy")]
    if torres_enemigas:
        torre = torres_enemigas[0]
        return (torre.x_norm, torre.y_norm)

    return (0.5, 0.3)

# Diccionario que asocia cada carta con su funcion de valor
FUNCIONES_DE_VALOR = {
    "giant": val_giant,
    "knight": val_knight,
    "musketeer": val_musketeer,
    "arrows": val_arrows,
    "fireball": val_fireball,
    "minipekka": val_minipekka,
    "minions": val_minions,
    "archers": val_archers
}

COSTES_CARTAS = {
    "giant": 5,
    "knight": 3,
    "musketeer": 4,
    "arrows": 3,
    "fireball": 4,
    "minipekka": 4,
    "minions": 3,
    "archers": 3,
}

def decidir_jugada(estado: GameState):
    mejor_carta = None
    mejor_valor = valor_esperar(estado)

    for nombre_carta in estado.mano:
        if nombre_carta is None or nombre_carta not in FUNCIONES_DE_VALOR:
            continue

        # NUEVO: si no tengo elixir suficiente, ni siquiera la evaluamos
        coste = COSTES_CARTAS.get(nombre_carta)
        if coste is None or estado.elixir < coste:
            continue

        valor = FUNCIONES_DE_VALOR[nombre_carta](estado)

        if valor > mejor_valor:
            mejor_valor = valor
            mejor_carta = nombre_carta

    if mejor_carta is None:
        return None

    return {"carta": mejor_carta, "valor": mejor_valor}


# Límites del área donde se puede desplegar (en y_norm), según tu hud_config.json
DEPLOY_Y_MIN = 0.48   # aprox. el puente
DEPLOY_Y_MAX = 0.78   # justo antes de la zona de cartas

PASO_REJILLA = 0.08   # cuanto mas pequeño, mas candidatos (mas preciso, mas lento)

DEPLOY_Y_MIN_INVASION = 0.2  # mucho mas adelante, dentro del campo enemigo


def generar_posiciones_candidatas(torres_destruidas=None):
    torres_destruidas = torres_destruidas or {"izquierda": False, "derecha": False}
    candidatas = []
    x = 0.10
    while x <= 0.90:
        lado = "izquierda" if x < 0.5 else "derecha"
        y_min = DEPLOY_Y_MIN_INVASION if torres_destruidas[lado] else DEPLOY_Y_MIN

        y = y_min
        while y <= DEPLOY_Y_MAX:
            candidatas.append((round(x, 2), round(y, 2)))
            y += PASO_REJILLA
        x += PASO_REJILLA
    return candidatas

def tropa_aliada_mas_adelantada(estado: GameState):
    """La tropa aliada con menor y_norm (mas cerca del puente/enemigo)."""
    if not estado.tropas_aliadas:
        return None
    return min(estado.tropas_aliadas, key=lambda t: t.y_norm)


def tropa_aliada_mas_atrasada(estado: GameState):
    """La tropa aliada con mayor y_norm (mas cerca de mi torre)."""
    if not estado.tropas_aliadas:
        return None
    return max(estado.tropas_aliadas, key=lambda t: t.y_norm)

def tanque_o_minitanque_aliado_mas_atrasado(estado: GameState):
    """La tropa aliada (solo tanque o minitanque) con mayor y_norm
    (la mas cerca de mi torre). Devuelve None si no hay ninguna."""
    candidatas = [t for t in estado.tropas_aliadas if t.tipo in (TANQUES | MINITANQUES)]
    if not candidatas:
        return None
    return max(candidatas, key=lambda t: t.y_norm)

def air_counter_aliado_mas_atrasado(estado: GameState):
    """La tropa aliada (solo tanque o minitanque) con mayor y_norm
    (la mas cerca de mi torre). Devuelve None si no hay ninguna."""
    candidatas = [t for t in estado.tropas_aliadas if t.tipo in (AIR_COUNTERS | MINITANQUES)]
    if not candidatas:
        return None
    return max(candidatas, key=lambda t: t.y_norm)


def tanque_aliado_en_campo(estado: GameState):
    return next((t for t in estado.tropas_aliadas if t.tipo in TANQUES), None)


def tropa_enemiga_mas_adelantada_en_mi_lado(estado: GameState):
    """La tropa enemiga mas invadida en mi lado (mayor y_norm)."""
    candidatas = [t for t in estado.tropas_enemigas if t.y_norm > MITAD_CAMPO_Y]
    if not candidatas:
        return None
    return max(candidatas, key=lambda t: t.y_norm)

def tropas_aliadas_atacando(estado: GameState):
    """Tropas aliadas que ya cruzaron al campo enemigo."""
    return [t for t in estado.tropas_aliadas if t.y_norm < MITAD_CAMPO_Y]

def puntuar_campo_vacio(pos, estado: GameState, nombre_carta=None) -> float:
    if estado.tropas_enemigas or estado.tropas_aliadas:
        return 0.0  # solo aplica si el campo esta completamente vacio

    x, y = pos
    distancia_centro = abs(x - 0.5)
    distancia_atras = DEPLOY_Y_MAX - y  # 0 cuando y = DEPLOY_Y_MAX (lo mas atras posible)

    return max(0.0, 1.0 - distancia_centro * 2) + max(0.0, 1.0 - distancia_atras * 3)


def puntuar_tanque(pos, estado: GameState, nombre_carta=None) -> float:
    if nombre_carta not in CARTAS_TANQUE:
        return 0.0

    # Si el campo esta completamente vacio, que decida puntuar_campo_vacio,
    # no esta funcion (evita competir por una posicion "por defecto")
    if not estado.tropas_enemigas and not estado.tropas_aliadas:
        return 0.0

    x, y = pos

    amenaza_avanzada = any(
        t.tipo in (AIR_COUNTERS_ENEMY | MINITANQUES_ENEMY) and t.y_norm > 0.50
        for t in estado.tropas_enemigas
    )
    if amenaza_avanzada:
        amenaza = tropa_enemiga_mas_adelantada_en_mi_lado(estado)
        objetivo_x = amenaza.x_norm if amenaza else 0.5
        distancia = math.hypot(x - objetivo_x, y - DEPLOY_Y_MIN)
        return max(0.0, 1.0 - distancia * 3)

    frontmost = air_counter_aliado_mas_atrasado(estado)
    if frontmost is None:
        objetivo_x, objetivo_y = 0.5, DEPLOY_Y_MAX  # sin tropas propias, detras (no en el puente)
    elif frontmost.y_norm > 0.5:
        objetivo_x = frontmost.x_norm
        objetivo_y = max(DEPLOY_Y_MIN, frontmost.y_norm - OFFSET_DETRAS_TROPA)
    else:
        objetivo_x, objetivo_y = 0.5, DEPLOY_Y_MAX  # sin tropas propias, detras (no en el puente)

    distancia = math.hypot(x - objetivo_x, y - objetivo_y)
    return max(0.0, 1.0 - distancia * 4)


def puntuar_minitanque(pos, estado: GameState, nombre_carta=None) -> float:
    if nombre_carta not in CARTAS_MINITANQUE:
        return 0.0
    if not estado.tropas_enemigas and not estado.tropas_aliadas:
        return 0.0
    x, y = pos

    enemigo = tropa_enemiga_mas_adelantada_en_mi_lado(estado)
    if enemigo is not None:
        objetivo_x = enemigo.x_norm
        objetivo_y = min(DEPLOY_Y_MAX, enemigo.y_norm + OFFSET_DETRAS_TROPA)
        distancia = math.hypot(x - objetivo_x, y - objetivo_y)
        return max(0.0, 1.0 - distancia * 3) * 2

    peso_cercania = 0.5 if tropas_aliadas_atacando(estado) else 1.0
    opciones = []
    tanque = tanque_aliado_en_campo(estado)
    if tanque is not None:
        objetivo_y = min(DEPLOY_Y_MAX, tanque.y_norm + OFFSET_DETRAS_TROPA)
        distancia = math.hypot(x - tanque.x_norm, y - objetivo_y)  # ahora usa x tambien
        opciones.append(max(0.0, 1.0 - distancia * 4) * peso_cercania)
        
    frontmost = tropa_aliada_mas_adelantada(estado)

    if frontmost is not None:
        objetivo_y = max(DEPLOY_Y_MIN, frontmost.y_norm - OFFSET_DETRAS_TROPA)
        distancia = math.hypot(x - frontmost.x_norm, y - objetivo_y)  # ahora usa x tambien
        opciones.append(max(0.0, 1.0 - distancia * 4) * peso_cercania)

    return max(opciones) if opciones else 0.0


def puntuar_air_counter(pos, estado: GameState, nombre_carta=None) -> float:
    if nombre_carta not in CARTAS_AIR_COUNTER:
        return 0.0
    if not estado.tropas_enemigas and not estado.tropas_aliadas:
        return 0.0
    x, y = pos
    amenaza = tropa_enemiga_mas_adelantada_en_mi_lado(estado)

    if amenaza is not None:
        offset = PASO_REJILLA * 2  # "2x2 casillas"
        # Hacia el centro del mapa: en x, acercarse a 0.5; en y, acercarse al puente
        objetivo_x = amenaza.x_norm + offset if amenaza.x_norm < 0.5 else amenaza.x_norm - offset
        objetivo_y = max(DEPLOY_Y_MIN, amenaza.y_norm + offset)
        distancia = math.hypot(x - objetivo_x, y - objetivo_y)
        return max(0.0, 1.0 - distancia * 4)

    # Sin enemigos en mi campo: detras del resto de mis tropas
    atras = tanque_o_minitanque_aliado_mas_atrasado(estado)
    if atras is not None:
        objetivo_x = atras.x_norm
        objetivo_y = min(DEPLOY_Y_MAX, atras.y_norm + PASO_REJILLA)
        distancia = math.hypot(x - objetivo_x, y - objetivo_y)
        return max(0.0, 1.0 - distancia * 4)
        #return max(0.0, 1.0 - abs(y - objetivo_y) * 4)

    return 0.0

FUNCIONES_POSICION = [
    puntuar_campo_vacio,
    puntuar_tanque,
    puntuar_minitanque,
    puntuar_air_counter,
]


def calcular_posicion_despliegue(estado: GameState, nombre_carta: str = None, torres_destruidas=None):
    candidatas = generar_posiciones_candidatas(torres_destruidas)
    mejor_posicion = None
    mejor_puntuacion = float("-inf")

    for pos in candidatas:
        puntuacion = sum(funcion(pos, estado, nombre_carta) for funcion in FUNCIONES_POSICION)
        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            mejor_posicion = pos

    return mejor_posicion

# ---------- BUCLE PRINCIPAL ----------

def main():
    print(f"Cargando modelo YOLO desde: {MODEL_PATH}")
    modelo_yolo = YOLO(MODEL_PATH)
    print(f"Modelo cargado con {len(modelo_yolo.names)} clases.")

    hud_config = cargar_hud_config()
    plantillas = cargar_plantillas()
    if plantillas:
        print(f"Cargadas {len(plantillas)} plantillas de cartas: {', '.join(plantillas.keys())}")
    else:
        print("Aviso: no hay plantillas de cartas en 'templates/', la mano saldra vacia.")

    sct = mss()
    plantilla_rematch = None
    if os.path.exists(TEMPLATE_REMATCH_PATH):
        plantilla_rematch = cv2.imread(TEMPLATE_REMATCH_PATH)
        print(f"Plantilla de 'Jugar de Nuevo' cargada desde: {TEMPLATE_REMATCH_PATH}")
    else:
        print(f"Aviso: no se encontro '{TEMPLATE_REMATCH_PATH}', el auto-rematch estara desactivado.")
    region = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE) or FALLBACK_REGION
    print(f"Capturando region inicial: {region}")

    frame_count = 0
    prev_time = time.time()
    ultima_jugada_timestamp = 0
    torre_tracker = TorreTracker()

    while True:
        if frame_count % REFRESH_WINDOW_EVERY_N_FRAMES == 0:
            nueva = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE)
            if nueva:
                region = nueva

        frame = capturar_frame(sct, region)

        # NUEVO: comprobar el boton de rematch cada N frames
        if plantilla_rematch is not None and frame_count % REVISAR_REMATCH_CADA_N_FRAMES == 0:
            encontrado, cx, cy, score = buscar_boton_rematch(frame, plantilla_rematch)
            if encontrado:
                x_absoluto = region["left"] + cx
                y_absoluto = region["top"] + cy
                print(f"[rematch] Boton detectado (score={score:.2f}), clic en ({x_absoluto}, {y_absoluto})")
                pyautogui.click(x_absoluto, y_absoluto)
                time.sleep(2)  # da tiempo a que cargue la siguiente partida
                frame_count += 1
                continue  # salta el resto de la iteracion (no tiene sentido jugar cartas en la pantalla de resultados)

        estado, resultados_yolo = construir_game_state(frame, modelo_yolo, hud_config, plantillas)

        torre_tracker.actualizar(estado.tropas)

        ahora = time.time()
        decision = None
        if ahora - ultima_jugada_timestamp > COOLDOWN_SEGUNDOS:
            decision = decidir_jugada(estado)

            if decision is not None:
                if decision["carta"] in HECHIZOS:
                    posicion = calcular_posicion_objetivo_hechizo(decision["carta"], estado)
                else:
                    posicion = calcular_posicion_despliegue(estado, decision["carta"], torre_tracker.destruida)

                jugado = jugar_carta(decision["carta"], estado.mano, hud_config, region, posicion)
                if jugado:
                    ultima_jugada_timestamp = ahora

        print(estado.resumen())

        # Visualizacion: dibuja las cajas de YOLO + info de HUD encima
        vista = resultados_yolo[0].plot()
        cv2.putText(vista, f"Elixir: {estado.elixir}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(vista, f"Mano: {estado.mano}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(vista, f"FPS: {fps:.1f}", (10, vista.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(PREVIEW_WINDOW_TITLE, vista)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
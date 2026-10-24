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

import cv2
import sys
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
MODEL_PATH = r"C:\DragonAI\runs\detect\train-13\weights\best.pt"
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

SWARMS_ENEMY = {"goblins_enemy", "spear_goblins_enemy", "bomber_enemy", "larry_enemy"}

SWARMS_FLY_ENEMY = {"minions_enemy", "bat_enemy"}

TROPAS_OBJETIVO_FIREBALL = {"musketeer_enemy", "barbarian_enemy", "squeleton_dragon_enemy", "wizard_enemy", "witch_enemy", "three_musketeer_enemy", "furnace_enemy", "hunter_enemy", "zappie_enemy"}

TROPAS_OBJETIVO_FLECHAS = {"minions_enemy", "archers_enemy", "bomber_enemy", "dart_goblin_enemy", "bat_enemy"}

RANGE_TROOPS_ENEMY = {"archers_enemy", "musketeer_enemy", "wizard_enemy", "witch_enemy", "three_musketeer_enemy", "bomber_enemy", "dart_goblin_enemy", "furnace_enemy", "hunter_enemy", "zappie_enemy"}

RANGE_FLY_ENEMY = {"baby_dragon_enemy", "flying_machine_enemy", "megaminion_enemy", "squeleton_dragon_enemy", "minions_enemy"}

TANQUES_ENEMY = {"giant_enemy", "royal_giant_enemy", "goblin_giant_enemy", "golem_enemy", "ice_golem_enemy", "rune_giant_enemy"}

BIG_TROOP_ENEMY = {"pekka_enemy", "squeleton_giant_enemy"}

WINCONS_ENEMY = {"battle_ram_enemy", "hog_rider_enemy", "piggie_enemy"}

WINCONS_FLY_ENEMY = {"balloon_enemy", "minion_giant_enemy", "squeleton_barrel_enemy"}

MINITANQUES_ENEMY = {"knight_enemy", "minipekka_enemy", "goblin_brawler_enemy", "valkyrie_enemy", "pekka_enemy", "dark_prince_enemy", "prince_enemy", "royal_recruit_enemy", "barbarian_enemy", "elite_barbarian_enemy", "guard_enemy", "healer_enemy", "berserker_enemy"}

ESTRUCUTAS_ENEMY = {"bomb_tower_enemy", "canon_enemy", "goblin_cage_enemy", "goblin_hut_enemy", "inferno_tower_enemy", "tombstone_enemy", "tesla_enemy"}

ESTRUCTURAS_OFFENSE_ENEMY = {"mortar_enemy","xbow_enemy"}

TANQUES = {"giant_player"}

MINITANQUES = {"knight_player", "minipekka_player", "prince_player", "dark_prince_player"}

AIR_COUNTERS = {"archers_player", "musketeer_player", "megaminion_player"}

MINITANQUES_AEREOS = {"megaminion_player"}

SPLASHERS = {"bomber_player"}

HECHIZOS = {"arrows", "fireball"}

RADIO_AGRUPACION_HECHIZO = {
    "arrows": 0.15,
    "fireball": 0.10,
}

CARTAS_TANQUE = {"giant"}

CARTAS_MINITANQUE = {"knight", "minipekka", "minions","prince", "dark_prince", "megaminion"}

CARTAS_AIR_COUNTER = {"musketeer", "archers", "bomber"}

OFFSET_DETRAS_TROPA = 0.08

TEMPLATE_REMATCH_PATH = r"C:\DragonAI\templates\jugar_de_nuevo.png"
UMBRAL_MATCH_REMATCH = 0.55
REVISAR_REMATCH_CADA_N_FRAMES = 15

RADIO_CERCANIA_ALIADA_HECHIZO = 0.11

PESO_OBJETIVO_DEFECTO = 2.0
PESO_COLATERAL = 0.5

# Puntuacion minima para que merezca la pena lanzar, y escala para val_*
SCORE_MINIMO_HECHIZO = {"fireball": 2.5, "arrows": 3.0}
SCORE_ESCALA_HECHIZO = {"fireball": 6.0, "arrows": 6.0}
# =========================================


TEMPLATE_INICIO_PATH = r"C:\DragonAI\templates\logo_inicio.png"
UMBRAL_MATCH_INICIO = 0.55
REVISAR_INICIO_CADA_N_FRAMES = 8   # el logo dura poco, no conviene revisarlo cada 15
COOLDOWN_INICIO_PARTIDA = 5        # segundos

TEMPLATE_X2_PATH = r"C:\DragonAI\templates\elixir_x2.png"
UMBRAL_MATCH_X2 = 0.55
REVISAR_X2_CADA_N_FRAMES = 15
MULTIPLICADOR_ELIXIR_X1 = 0.6   # valor de las cartas fuera de x2
SEGUNDOS_PERSISTENCIA_X2 = 3 

# ---------- ESTRUCTURA DE DATOS ----------

@dataclass
class Tropa:
    tipo: str
    x_norm: float
    y_norm: float
    confianza: float
    w_norm: float = 0.0
    h_norm: float = 0.0

    @property
    def y_pie(self) -> float:
        return self.y_norm + self.h_norm / 2


@dataclass
class GameState:
    timestamp: float
    elixir: float
    mano: list  # lista de 4 nombres de carta (o None si no se reconoce)
    tropas: list = field(default_factory=list)  # lista de objetos Tropa
    vida_torres: dict = field(default_factory=dict)        # {"player_izq": 0.6, ...}
    torres_destruidas: dict = field(default_factory=dict)  # el de TorreTracker

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
    """Carga las plantillas agrupadas por carta: {nombre_carta: [imagenes]}.
    Los archivos con doble guion bajo (ej. archers__evo.png) se tratan como
    variantes de la carta que va antes del '__' (en este caso 'archers')."""
    plantillas = {}
    if not os.path.isdir(TEMPLATES_DIR):
        return plantillas
    for archivo in os.listdir(TEMPLATES_DIR):
        if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
            nombre_archivo = os.path.splitext(archivo)[0]
            nombre_carta = nombre_archivo.split("__")[0]
            img = cv2.imread(os.path.join(TEMPLATES_DIR, archivo))
            if img is not None:
                plantillas.setdefault(nombre_carta, []).append(img)
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
    for nombre, lista_plantillas in plantillas.items():
        for plantilla in lista_plantillas:
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
    FRAMES_CONFIRMAR_VISTA = 10
    FRAMES_AUSENTE_PARA_DESTRUIDA = 30

    def __init__(self, tipo="princess_tower_enemy"):
        self.tipo = tipo
        self.vista_consecutiva = {"izquierda": 0, "derecha": 0}
        self.ausente_consecutiva = {"izquierda": 0, "derecha": 0}
        self.confirmada = {"izquierda": False, "derecha": False}
        self.destruida = {"izquierda": False, "derecha": False}

    def actualizar(self, tropas):
        vista_este_frame = {"izquierda": False, "derecha": False}
        for t in tropas:
            if t.tipo == self.tipo:   # antes: "princess_tower_enemy" fijo
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
                        print(f"[torres] {self.tipo} {lado.upper()} destruida")



DIFF_SLOT_MAX = 8            # diferencia media (0-255) para considerar que no cambio
REFRESCAR_SLOT_CADA = 30     # fuerza un reconocimiento cada N llamadas por slot

_cache_slots = {}  
def identificar_carta_cacheada(i, slot_img, plantillas):
    if slot_img is None or slot_img.size == 0:
        return None
    mini = cv2.resize(slot_img, (16, 20), interpolation=cv2.INTER_AREA)
    previo = _cache_slots.get(i)
    if previo is not None:
        previo["edad"] += 1
        igual = np.mean(cv2.absdiff(mini, previo["mini"])) < DIFF_SLOT_MAX
        if igual and previo["edad"] < REFRESCAR_SLOT_CADA:
            return previo["nombre"]
    nombre, _ = identificar_carta(slot_img, plantillas)
    _cache_slots[i] = {"mini": mini, "nombre": nombre, "edad": 0}
    return nombre
# ---------- CONSTRUCCION DEL GAMESTATE ----------

def construir_game_state(frame, modelo_yolo, hud_config, plantillas):
    h, w = frame.shape[:2]

    t0 = time.perf_counter()
    # --- Elixir ---
    elixir = leer_elixir(frame, tuple(hud_config["elixir_barra"]), w, h)
    crono.medir("elixir", t0)

    t0 = time.perf_counter()
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
        nombre = identificar_carta_cacheada(i, recorte, plantillas)
        mano.append(nombre)
    crono.medir("cartas", t0)

    t0 = time.perf_counter()
    # --- Tropas/torres (YOLO) ---
    resultados = modelo_yolo.predict(source=frame, conf=CONFIDENCE_THRESHOLD_YOLO, verbose=False, device=0)
    tropas = []
    for box in resultados[0].boxes:
        cls_id = int(box.cls[0])
        tipo = modelo_yolo.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # centro de la caja
        y_ref = cy if tipo in TORRES else y2   # y2 = borde inferior de la caja
        tropas.append(Tropa(tipo=tipo, x_norm=round(cx / w, 3), y_norm=round(y_ref / h, 3),confianza=round(conf, 2),w_norm=round((x2 - x1) / w, 3), h_norm=round((y2 - y1) / h, 3),))
    crono.medir("tropas", t0)
    return GameState(timestamp=time.time(), elixir=elixir, mano=mano, tropas=tropas), resultados


# ---------- TORRES ----------
# OpenCV usa H en 0-179 (grados / 2), S y V en 0-255
BARRA_HSV = {
    "enemy":  {"h": (165, 177), "s_min": 100, "v_min": 150},
    "player": {"h": (95, 106),  "s_min": 100, "v_min": 150},
}
ANCHO_BARRA_NORM = 0.10      # ancho de la barra completa / ancho de ventana
BARRA_PX_MINIMOS = 3


def clave_torre(t):
    bando = "enemy" if t.tipo.endswith("_enemy") else "player"
    if t.tipo.startswith("king"):
        return f"{bando}_king"
    return f"{bando}_{'izq' if t.x_norm < 0.5 else 'der'}"


# (desde, hasta) como fraccion del alto de la caja, medidos desde su borde superior
BARRA_Y_ENEMY = (-0.17, -0.05)   # encima de la torre
BARRA_Y_PLAYER = (0.64, 0.76)    # dentro, en la parte baja
BARRA_DESPLAZAMIENTO_X = 0.015   # fraccion del ancho de la ventana; positivo = derecha

def region_barra(t, w, h):
    desde, hasta = BARRA_Y_ENEMY if t.tipo.endswith("_enemy") else BARRA_Y_PLAYER
    cx = (t.x_norm + BARRA_DESPLAZAMIENTO_X) * w   # antes: t.x_norm * w
    alto = t.h_norm * h
    top = t.y_norm * h - alto / 2
    mitad = ANCHO_BARRA_NORM * w / 2 * 1.05
    x1, x2 = int(max(0, cx - mitad)), int(min(w, cx + mitad))
    y1 = int(max(0, top + desde * alto))
    y2 = int(min(h, top + hasta * alto))
    return x1, y1, x2, y2


def leer_vida_torre(frame, t):
    """Fraccion 0-1 de vida, o None si no se ve barra (torre intacta)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region_barra(t, w, h)
    if x2 <= x1 or y2 <= y1:
        return None

    bando = "enemy" if t.tipo.endswith("_enemy") else "player"
    cfg = BARRA_HSV[bando]
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    mascara = (
        (hsv[..., 0] >= cfg["h"][0]) & (hsv[..., 0] <= cfg["h"][1])
        & (hsv[..., 1] >= cfg["s_min"]) & (hsv[..., 2] >= cfg["v_min"])
    )

    cols = np.where(np.any(mascara, axis=0))[0]
    if len(cols) < BARRA_PX_MINIMOS:
        return None

    relleno = cols.max() - cols.min() + 1
    return float(min(1.0, relleno / (x2 - x1)))


class VidaTorresTracker:
    def __init__(self):
        self.historial = {}
        self.vida = {}

    def actualizar(self, frame, tropas, destruidas_enemy=None, destruidas_player=None):
        for t in tropas:
            if t.tipo not in TORRES or t.tipo.startswith("king"):
                continue
            k = clave_torre(t)
            lectura = leer_vida_torre(frame, t)
            if lectura is None:
                continue
            hist = self.historial.setdefault(k, [])
            hist.append(lectura)
            del hist[:-5]
            mediana = float(sorted(hist)[len(hist) // 2])
            self.vida[k] = float(min(self.vida.get(k, 1.0), mediana))

        # Torre destruida = vida 0, aunque ya no se lea ninguna barra
        for lado, clave in (("izquierda", "enemy_izq"), ("derecha", "enemy_der")):
            if destruidas_enemy and destruidas_enemy.get(lado):
                self.vida[clave] = 0.0
        for lado, clave in (("izquierda", "player_izq"), ("derecha", "player_der")):
            if destruidas_player and destruidas_player.get(lado):
                self.vida[clave] = 0.0

        return dict(self.vida)


# ---------- LOGICA DE CARTAS BASICA ----------

def hay_aliada_cerca(centro, estado: GameState, radio=RADIO_CERCANIA_ALIADA_HECHIZO) -> bool:
    """Comprueba si alguna tropa aliada esta dentro de 'radio' del centro dado."""
    x_centro, y_centro = centro
    return any(
        math.hypot(t.x_norm - x_centro, t.y_norm - y_centro) < radio
        for t in estado.tropas_aliadas
    )

def ventaja_tropas(estado: GameState) -> float:
    """Devuelve de 0 a 1 quien tiene mas tropas en campo (sin torres).
    0.0 = solo tiene tropas el rival, 0.5 = igualdad, 1.0 = solo tengo yo.
    Si no hay ninguna tropa en campo, devuelve 0.5 (neutral)."""
    mias = len(estado.tropas_aliadas)
    rivales = len(estado.tropas_enemigas)
    total = mias + rivales

    if total == 0:
        return 0.5

    return round(mias / total, 2)

def hay_en_mi_lado(tropas, tipos, y_min=MITAD_CAMPO_Y):
    """True si alguna tropa de 'tipos' esta en mi lado (y_norm > y_min)."""
    return any(t.tipo in tipos and t.y_norm > y_min for t in tropas)


def hay_en_campo(tropas, tipos):
    """True si hay alguna tropa de 'tipos' en cualquier parte."""
    return any(t.tipo in tipos for t in tropas)

# Que tropas justifican lanzar cada hechizo
OBJETIVOS_HECHIZO = {
    "fireball": TROPAS_OBJETIVO_FIREBALL,
    "arrows": TROPAS_OBJETIVO_FLECHAS | SWARMS_ENEMY,
}

# Peso de cada tropa dentro del area. Objetivo prioritario = peso alto.
PESO_OBJETIVO_HECHIZO = {
    "fireball": {
        "witch_enemy": 2.5,
        "three_musketeer_enemy": 2.5,
        "wizard_enemy": 2.5,
        "musketeer_enemy": 2,
        "squeleton_dragon_enemy": 1.5,
        "barbarian_enemy": 1,
    },
    "arrows": {
        "minions_enemy": 1.5,
        "archers_enemy": 1.5,
        "bomber_enemy": 1.5,
        "goblins_enemy": 1,
        "spear_goblins_enemy": 1,
        "larry_enemy": 0.5,
        "bat_enemy": 1.0,
    },
}

def peso_tropa_hechizo(tropa, nombre_carta):
    if tropa.tipo in OBJETIVOS_HECHIZO.get(nombre_carta, set()):
        return PESO_OBJETIVO_HECHIZO.get(nombre_carta, {}).get(tropa.tipo, PESO_OBJETIVO_DEFECTO)
    return PESO_COLATERAL

DISTANCIA_MINIMA_KING_TOWER = 0.2
OFFSET_Y_HECHIZO = 0.06


def punto_lanzamiento_hechizo(centro, estado: GameState):
    """Punto real donde caeria el hechizo: sin offset si hay aliadas
    cerca (para no fallar el area), con offset si no las hay."""
    x, y = centro
    if not hay_aliada_cerca(centro, estado):
        y += OFFSET_Y_HECHIZO
    return (x, max(0.0, min(1.0, y)))


def demasiado_cerca_king_tower(punto, estado: GameState) -> bool:
    """True si el punto esta a menos de DISTANCIA_MINIMA_KING_TOWER
    de la king tower enemiga. Si no se ve la torre, no restringe."""
    return any(
        math.hypot(punto[0] - t.x_norm, punto[1] - t.y_norm) < DISTANCIA_MINIMA_KING_TOWER
        for t in estado.tropas if t.tipo == "king_tower_enemy"
    )


def mejor_objetivo_hechizo(nombre_carta, estado: GameState):
    """Devuelve (puntuacion, punto_lanzamiento). Solo las tropas objetivo
    pueden ser ancla, y se descartan los puntos cerca de la king tower."""
    objetivos = OBJETIVOS_HECHIZO.get(nombre_carta, set())
    radio = RADIO_AGRUPACION_HECHIZO.get(nombre_carta, 0.12)
    tropas = estado.tropas_enemigas  # ya sin filtro de y: todo el campo

    mejor_score, mejor_punto = 0.0, None
    for ancla in tropas:
        if ancla.tipo not in objetivos:
            continue

        centro = (ancla.x_norm, ancla.y_norm)
        punto = punto_lanzamiento_hechizo(centro, estado)
        if demasiado_cerca_king_tower(punto, estado):
            continue

        grupo = [
            t for t in tropas
            if math.hypot(t.x_norm - ancla.x_norm, t.y_norm - ancla.y_norm) < radio
        ]
        score = sum(peso_tropa_hechizo(t, nombre_carta) for t in grupo)
        if score > mejor_score:
            mejor_score = score
            mejor_punto = punto

    return round(mejor_score, 2), mejor_punto

def val_giant(estado: GameState):
    valor = 0.0

    if estado.elixir > 8:
        valor += 0.2

    # Ventaja numerica: +0.1 si voy ganando, -0.1 si voy perdiendo
    ventaja = ventaja_tropas(estado)
    if ventaja > 0.5:
        valor += 0.1
    elif ventaja < 0.5:
        valor -= 0.1

    # +0.1 si tengo antiaereos aliados en mi lado
    if hay_en_mi_lado(estado.tropas_aliadas, AIR_COUNTERS):
        valor += 0.1

    # +0.1 si tengo minitanques aliados en mi lado
    if hay_en_mi_lado(estado.tropas_aliadas, MINITANQUES):
        valor += 0.1

    # +0.2 si tengo un principe aliado en mi lado
    if hay_en_mi_lado(estado.tropas_aliadas, {"prince_player"}):
        valor += 0.2

    # -0.2 si hay minitanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, MINITANQUES_ENEMY):
        valor -= 0.2

    # -0.2 si hay winconditions enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, WINCONS_ENEMY):
        valor -= 0.2

    # -0.2 si hay pekkas enemigos en mi lado
    if hay_en_campo(estado.tropas_enemigas, BIG_TROOP_ENEMY):
        valor -= 0.2

    return round(max(0.0, min(valor, 1.0)), 2)

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
        t.tipo in (RANGE_TROOPS_ENEMY) and t.y_norm > MITAD_CAMPO_Y-0.1
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

VALOR_MAXIMO_HECHIZO = {"fireball": 0.7, "arrows": 0.6}

def val_hechizo(nombre_carta, estado: GameState):
    score, _ = mejor_objetivo_hechizo(nombre_carta, estado)
    if score < SCORE_MINIMO_HECHIZO[nombre_carta]:
        return 0.0
    valor = min(score / SCORE_ESCALA_HECHIZO[nombre_carta], 1.0)
    return round(valor * VALOR_MAXIMO_HECHIZO[nombre_carta], 2)

def val_arrows(estado: GameState):
    return val_hechizo("arrows", estado)

def val_fireball(estado: GameState):
    return val_hechizo("fireball", estado)

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
        t.tipo in RANGE_TROOPS_ENEMY and t.y_norm > MITAD_CAMPO_Y
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

    if any(t.tipo in RANGE_TROOPS_ENEMY for t in estado.tropas_enemigas):
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

    # +0.1 si hay tanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, TANQUES_ENEMY):
        valor += 0.1

    # +0.2 si hay minitanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, MINITANQUES_ENEMY):
        valor += 0.2

    # +0.1 si hay swarms enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, SWARMS_ENEMY):
        valor += 0.1

    # +0.1 si hay tanques aliados en mi lado
    if hay_en_mi_lado(estado.tropas_aliadas, TANQUES):
        valor += 0.1
    
    # +0.1 si hay wincons enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, WINCONS_ENEMY):
        valor += 0.1
    
    # -0.2 si hay un mago enemigo en campo
    if hay_en_campo(estado.tropas_aliadas, {"wizard_enemy"}):
        valor -= 0.2

    # +0.3 si hay wincons voladoras enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, WINCONS_FLY_ENEMY):
        valor += 0.3

    # +0.1 si hay wincons voladoras enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, RANGE_FLY_ENEMY):
        valor += 0.1

    # +0.2 si hay pekkas enemigos en mi lado
    if hay_en_campo(estado.tropas_enemigas, BIG_TROOP_ENEMY):
        valor += 0.2

    # -0.3 si hay swarms aereas enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, SWARMS_FLY_ENEMY):
        valor += 0.3
    

    return round(min(valor, 1.0), 2)


def val_bomber(estado: GameState):
    valor = 0.1

    # +0.2 si hay swarms enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, SWARMS_ENEMY):
        valor += 0.2

    # +0.2 si hay swarms enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, SWARMS_ENEMY):
        valor += 0.2 
    
    # +0.1 si hay tanques aliados en mi lado
    if hay_en_mi_lado(estado.tropas_aliadas, TANQUES):
        valor += 0.1
    
    # +0.1 si hay wincons enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, WINCONS_ENEMY):
        valor += 0.1

    # -0.1 si hay tropas voladoras enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, RANGE_FLY_ENEMY):
        valor -= 0.1

    # +0.2 si hay pekkas enemigos en mi lado
    if hay_en_campo(estado.tropas_enemigas, BIG_TROOP_ENEMY):
        valor += 0.2

    # +0.1 si hay minitanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, MINITANQUES_ENEMY):
        valor += 0.1

    # -0.3 si hay swarms aereas enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, SWARMS_FLY_ENEMY):
        valor -= 0.3


    return round(min(valor, 1.0), 2)


def val_prince(estado: GameState):
    valor = 0.1

    # +0.3 si hay tanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, TANQUES_ENEMY):
        valor += 0.3

    # +0.2 si hay minitanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, MINITANQUES_ENEMY):
        valor += 0.2

    # +0.3 si hay tropas de rango enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, RANGE_TROOPS_ENEMY):
        valor += 0.3

    # -0.2 si hay swarms enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, SWARMS_ENEMY):
        valor -= 0.2

    # -0.2 si hay swarms enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, SWARMS_ENEMY):
        valor -= 0.2
    
    # +0.2 si hay tanques aliados en mi lado
    if hay_en_mi_lado(estado.tropas_aliadas, TANQUES):
        valor += 0.2

    # +0.2 si hay tanques aliados en el campo
    if hay_en_campo(estado.tropas_aliadas, TANQUES):
        valor += 0.2
    
    # +0.3 si hay wincons enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, WINCONS_ENEMY):
        valor += 0.3
    
    # -0.2 si hay wincons voladoras enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, RANGE_FLY_ENEMY):
        valor -= 0.2
    
    # +0.1 si hay pekkas enemigos en mi lado
    if hay_en_campo(estado.tropas_enemigas, BIG_TROOP_ENEMY):
        valor += 0.1

    # -0.3 si hay swarms aereas enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, SWARMS_FLY_ENEMY):
        valor -= 0.3
    
    return round(min(valor, 1.0), 2)

def val_megaminion(estado: GameState):
    valor = 0.1

    # +0.2 si hay tanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, TANQUES_ENEMY):
        valor += 0.2

    # +0.3 si hay minitanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, MINITANQUES_ENEMY):
        valor += 0.3

    # +0.1 si hay tanques aliados en mi lado
    if hay_en_mi_lado(estado.tropas_aliadas, TANQUES):
        valor += 0.1
    
    # +0.2 si hay wincons enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, WINCONS_ENEMY):
        valor += 0.2
    
    # +0.5 si hay wincons voladoras enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, WINCONS_FLY_ENEMY):
        valor += 0.5

    # +0.2 si hay tropas voladoras enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, RANGE_FLY_ENEMY):
        valor += 0.2

    # +0.3 si hay pekkas enemigos en mi lado
    if hay_en_campo(estado.tropas_enemigas, BIG_TROOP_ENEMY):
        valor += 0.3

    # -0.2 si hay rangetroop enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, BIG_TROOP_ENEMY):
        valor -= 0.2

    # +0.3 si hay swarms aereas enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, SWARMS_FLY_ENEMY):
        valor += 0.1
    
    return round(min(valor, 1.0), 2)

def val_dark_prince(estado: GameState):
    valor = 0.1

    # +0.1 si hay tanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, TANQUES_ENEMY):
        valor += 0.1

    # +0.2 si hay minitanques enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, MINITANQUES_ENEMY):
        valor += 0.2

    # +0.2 si hay tropas de rango enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, RANGE_TROOPS_ENEMY):
        valor += 0.2

    # +0.2 si hay swarms enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, SWARMS_ENEMY):
        valor += 0.2

    # +0.1 si hay swarms enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, SWARMS_ENEMY):
        valor += 0.1
    
    # +0.1 si hay tanques aliados en mi lado
    if hay_en_mi_lado(estado.tropas_aliadas, TANQUES):
        valor += 0.1

    # +0.1 si hay tanques aliados en el campo
    if hay_en_campo(estado.tropas_aliadas, TANQUES):
        valor += 0.1
    
    # +0.2 si hay wincons enemigos en mi lado
    if hay_en_mi_lado(estado.tropas_enemigas, WINCONS_ENEMY):
        valor += 0.2
    
    # -0.2 si hay tropas voladoras enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, RANGE_FLY_ENEMY):
        valor -= 0.2
    
    # +0.2 si hay pekkas enemigos en mi lado
    if hay_en_campo(estado.tropas_enemigas, BIG_TROOP_ENEMY):
        valor += 0.2

    # -0.3 si hay swarms aereas enemigos en el campo
    if hay_en_campo(estado.tropas_enemigas, SWARMS_FLY_ENEMY):
        valor -= 0.3
    
    return round(min(valor, 1.0), 2)


def valor_esperar(estado: GameState):
    valor = max(0.0, 1.0 - estado.elixir / 10) * 0.6

    return round(max(0.0, min(valor, 1.0)), 2)


def torre_enemiga_objetivo(estado: GameState):
    torres = [t for t in estado.tropas if t.tipo == "princess_tower_enemy"]
    if not torres:
        return None
    return min(torres, key=lambda t: estado.vida_torres.get(clave_torre(t), 1.0))


def calcular_posicion_objetivo_hechizo(nombre_carta: str, estado: GameState):
    _, punto = mejor_objetivo_hechizo(nombre_carta, estado)
    if punto is not None:
        return punto

    # Fallback: princess tower enemiga (nunca la king tower)
    for t in estado.tropas:
        torre = torre_enemiga_objetivo(estado)
        if torre is not None:
            punto = (torre.x_norm, torre.y_norm)
            if not demasiado_cerca_king_tower(punto, estado):
                return punto

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
    "archers": val_archers,
    "bomber": val_bomber,
    "prince": val_prince,
    "megaminion": val_megaminion,
    "dark_prince": val_dark_prince
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
    "bomber": 2,
    "prince": 5,
    "megaminion": 3,
    "dark_prince": 4
}

def decidir_jugada(estado: GameState, multiplicador=1.0):
    mejor_carta = None
    mejor_valor = valor_esperar(estado)

    for nombre_carta in estado.mano:
        if nombre_carta is None or nombre_carta not in FUNCIONES_DE_VALOR:
            continue

        coste = COSTES_CARTAS.get(nombre_carta)
        if coste is None or estado.elixir < coste:
            continue

        valor = FUNCIONES_DE_VALOR[nombre_carta](estado) * multiplicador

        if valor > mejor_valor:
            mejor_valor = valor
            mejor_carta = nombre_carta

    if mejor_carta is None:
        return None

    return {"carta": mejor_carta, "valor": round(mejor_valor, 2)}


# Límites del área donde se puede desplegar (en y_norm), según tu hud_config.json
DEPLOY_Y_MIN = 0.48   # aprox. el puente
DEPLOY_Y_MAX = 0.78   # justo antes de la zona de cartas

PASO_REJILLA = 0.08   # cuanto mas pequeño, mas candidatos (mas preciso, mas lento)

DEPLOY_Y_MIN_INVASION = 0.2  # mucho mas adelante, dentro del campo enemigo

OFFSET_DETRAS_TANQUE = 0.06

Y_REPOSO = 0.70                                  # atras, sin llegar a las cartas
CARRIL_X = {"izquierda": 0.27, "derecha": 0.73}  # ajusta a tus princess towers


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

CARTAS_LADO_DEBIL = CARTAS_TANQUE | CARTAS_MINITANQUE   # cartas que van al lado debil
DIFERENCIA_VIDA_LADO_COMPLETO = 0.30   # con esta diferencia de vida va del todo al carril


def puntuar_campo_vacio(pos, estado: GameState, nombre_carta=None) -> float:
    if estado.tropas_enemigas or estado.tropas_aliadas:
        return 0.0  # solo aplica si el campo esta completamente vacio

    x, y = pos

    # Por defecto, al centro. Si hay una torre enemiga claramente mas debil,
    # el objetivo se desplaza hacia su carril, de forma gradual.
    objetivo_x = 0.5
    if nombre_carta in CARTAS_LADO_DEBIL:
        lado, diferencia = lado_enemigo_mas_debil(estado)
        if lado is not None:
            carril = CARRIL_X["izquierda" if lado == "izq" else "derecha"]
            peso = min(1.0, diferencia / DIFERENCIA_VIDA_LADO_COMPLETO)
            objetivo_x = 0.5 + (carril - 0.5) * peso

    distancia_centro = abs(x - objetivo_x)
    distancia_atras = DEPLOY_Y_MAX - y

    return max(0.0, 1.0 - distancia_centro * 2) + max(0.0, 1.0 - distancia_atras * 3)



DIFERENCIA_VIDA_CAMBIO_CARRIL = 0.20   # a partir de aqui se ataca el lado debil


def x_objetivo_reposo(estado: GameState, nombre_carta=None) -> float:
    """x donde colocar la carta atras cuando no hay enemigos en mi lado.
    Por defecto sigue el carril de la amenaza; si una princess enemiga esta
    claramente mas debil, tanques y minitanques van a ese carril."""
    x_amenaza = carril_de_amenaza(estado)
    if nombre_carta not in CARTAS_LADO_DEBIL:
        return x_amenaza

    lado, diferencia = lado_enemigo_mas_debil(estado)
    if lado is None or diferencia < DIFERENCIA_VIDA_CAMBIO_CARRIL:
        return x_amenaza

    return CARRIL_X["izquierda" if lado == "izq" else "derecha"]

def carril_de_amenaza(estado: GameState):
    """x del carril por el que viene el enemigo. Usa la tropa enemiga mas
    adelantada (mayor y_norm) y la ajusta al centro de su carril.
    Si no hay enemigos, devuelve el centro (0.5)."""
    if not estado.tropas_enemigas:
        return 0.5
    t = max(estado.tropas_enemigas, key=lambda t: t.y_norm)
    return CARRIL_X["izquierda"] if t.x_norm < 0.5 else CARRIL_X["derecha"]

def puntuar_posicion_reposo(pos, estado: GameState, nombre_carta=None) -> float:
    x, y = pos
    objetivo_x = x_objetivo_reposo(estado, nombre_carta)
    distancia = math.hypot(x - objetivo_x, y - Y_REPOSO)
    return max(0.0, 1.0 - distancia * 3)



PESO_ATAQUE_TORRE_DEBIL = 1.0
DIFERENCIA_MIN_VIDA = 0.10   # por debajo, no hay preferencia


def lado_enemigo_mas_debil(estado: GameState):
    """('izq'|'der', diferencia) o (None, 0).
    - Si cae una princess, el objetivo es la que sigue en pie.
    - Si caen las dos, no hay preferencia."""
    d = estado.torres_destruidas or {}
    izq_caida = d.get("izquierda", False)
    der_caida = d.get("derecha", False)

    if izq_caida and der_caida:
        return None, 0.0
    if izq_caida:
        return "der", 1.0
    if der_caida:
        return "izq", 1.0

    izq = estado.vida_torres.get("enemy_izq", 1.0)
    der = estado.vida_torres.get("enemy_der", 1.0)
    if abs(izq - der) < DIFERENCIA_MIN_VIDA:
        return None, 0.0
    return ("izq" if izq < der else "der"), abs(izq - der)


def puntuar_ataque_torre_debil(pos, estado: GameState, nombre_carta=None) -> float:
    if nombre_carta not in (CARTAS_TANQUE | CARTAS_MINITANQUE):
        return 0.0
    if tropa_enemiga_mas_adelantada_en_mi_lado(estado) is not None:
        return 0.0   # primero defender
    lado, diferencia = lado_enemigo_mas_debil(estado)
    if lado is None:
        return 0.0
    objetivo_x = CARRIL_X["izquierda" if lado == "izq" else "derecha"]
    cercania = max(0.0, 1.0 - abs(pos[0] - objetivo_x) * 3)
    return cercania * PESO_ATAQUE_TORRE_DEBIL * min(1.0, diferencia / 0.4)


# --- Constantes (junto a OFFSET_DETRAS_TROPA, etc.) ---
PESO_DEFENSA = 3.0                  # mayor que PESO_DETRAS_TANQUE (1.5)
UMBRAL_DEFENSA_Y = MITAD_CAMPO_Y    # a partir de aqui el enemigo cuenta como amenaza
URGENCIA_MAX = 1.0                  # con la torre casi muerta, la defensa pesa hasta x2

def factor_urgencia_defensa(estado: GameState, amenaza) -> float:
    """1.0 con mi torre del carril entera, hasta 1 + URGENCIA_MAX con la torre casi muerta."""
    lado = "izq" if amenaza.x_norm < 0.5 else "der"
    vida = estado.vida_torres.get(f"player_{lado}", 1.0)
    return 1.0 + URGENCIA_MAX * (1.0 - vida)


def puntuar_defensa(pos, estado: GameState, offset_y=OFFSET_DETRAS_TROPA, escala_distancia=2.0):
    """Puntuacion alta y estable de defender al enemigo mas adelantado en mi
    lado. No depende de lo cerca que este de mi torre: solo de que haya
    cruzado la mitad. Devuelve 0.0 si no hay amenaza."""
    amenazas = [t for t in estado.tropas_enemigas if t.y_norm > UMBRAL_DEFENSA_Y]
    if not amenazas:
        return 0.0

    amenaza = max(amenazas, key=lambda t: t.y_norm)
    x, y = pos
    objetivo_x = amenaza.x_norm
    objetivo_y = min(DEPLOY_Y_MAX, max(DEPLOY_Y_MIN, amenaza.y_norm + offset_y))
    distancia = math.hypot(x - objetivo_x, y - objetivo_y)
    factor = factor_urgencia_defensa(estado, amenaza)
    return max(0.0, 1.0 - distancia * escala_distancia) * PESO_DEFENSA * factor

def puntuar_tanque(pos, estado: GameState, nombre_carta=None) -> float:
    if nombre_carta not in CARTAS_TANQUE:
        return 0.0

    # Si el campo esta completamente vacio, que decida puntuar_campo_vacio,
    # no esta funcion (evita competir por una posicion "por defecto")
    if not estado.tropas_enemigas and not estado.tropas_aliadas:
        return 0.0

    x, y = pos

    amenaza_avanzada = any(
        t.tipo in (RANGE_TROOPS_ENEMY | MINITANQUES_ENEMY) and t.y_norm > 0.50
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
        objetivo_x, objetivo_y = x_objetivo_reposo(estado, nombre_carta), Y_REPOSO

    distancia = math.hypot(x - objetivo_x, y - objetivo_y)
    return max(0.0, 1.0 - distancia * 4)



PESO_DETRAS_TANQUE = 1.5

def puntuar_minitanque(pos, estado: GameState, nombre_carta=None) -> float:
    if nombre_carta not in CARTAS_MINITANQUE:
        return 0.0
    if not estado.tropas_enemigas and not estado.tropas_aliadas:
        return 0.0
    x, y = pos
    opciones = []

    # Opcion A: detras de mi tanque (prioritaria, sin penalizar por atacar)
    tanque = tanque_aliado_en_campo(estado)
    if tanque is not None:
        objetivo_y = min(DEPLOY_Y_MAX, max(DEPLOY_Y_MIN, tanque.y_norm + OFFSET_DETRAS_TANQUE))
        distancia = math.hypot(x - tanque.x_norm, y - objetivo_y)
        opciones.append(max(0.0, 1.0 - distancia * 4) * PESO_DETRAS_TANQUE)

    # Opcion B: defender
    defensa = puntuar_defensa(pos, estado)
    if defensa > 0:
        opciones.append(defensa)

    # Opcion C: detras de la tropa aliada mas adelantada
    frontmost = tropa_aliada_mas_adelantada(estado)
    if frontmost is not None:
        peso_cercania = 0.5 if tropas_aliadas_atacando(estado) else 1.0
        objetivo_y = max(DEPLOY_Y_MIN, frontmost.y_norm - OFFSET_DETRAS_TANQUE)
        distancia = math.hypot(x - frontmost.x_norm, y - objetivo_y)
        opciones.append(max(0.0, 1.0 - distancia * 4) * peso_cercania)

    if opciones:
        return max(opciones)
    return puntuar_posicion_reposo(pos, estado, nombre_carta)   # <-- AQUI


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

    return puntuar_posicion_reposo(pos, estado, nombre_carta)

FUNCIONES_POSICION = [
    puntuar_campo_vacio,
    puntuar_tanque,
    puntuar_minitanque,
    puntuar_air_counter,
    puntuar_ataque_torre_debil,
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

def depurar_barras(frame, tropas):
    h, w = frame.shape[:2]
    for t in tropas:
        if t.tipo not in TORRES or t.tipo.startswith("king"):
            continue
        x1, y1, x2, y2 = region_barra(t, w, h)
        if x2 <= x1 or y2 <= y1:
            print(f"[barra] {clave_torre(t)}: rectangulo vacio {(x1, y1, x2, y2)}")
            continue
        bando = "enemy" if t.tipo.endswith("_enemy") else "player"
        cfg = BARRA_HSV[bando]
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        mascara = (
            (hsv[..., 0] >= cfg["h"][0]) & (hsv[..., 0] <= cfg["h"][1])
            & (hsv[..., 1] >= cfg["s_min"]) & (hsv[..., 2] >= cfg["v_min"])
        )
        cols = int(np.any(mascara, axis=0).sum())
        print(f"[barra] {clave_torre(t)}: rect={(x1, y1, x2, y2)} "
              f"pix_mascara={int(mascara.sum())} columnas={cols} "
              f"hue_mediano={int(np.median(hsv[..., 0]))}")
        cv2.imwrite(f"debug_{clave_torre(t)}.png", frame[y1:y2, x1:x2])




class Cronometro:
    def __init__(self):
        self.acum = {}
        self.n = 0

    def medir(self, nombre, t0):
        self.acum[nombre] = self.acum.get(nombre, 0.0) + time.perf_counter() - t0

    def informar(self, cada=60):
        self.n += 1
        if self.n % cada == 0:
            print("[tiempos ms/frame]", {k: round(v / cada * 1000, 1) for k, v in self.acum.items()})
            self.acum.clear()

crono = Cronometro()
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

    plantilla_inicio = None
    if os.path.exists(TEMPLATE_INICIO_PATH):
        plantilla_inicio = cv2.imread(TEMPLATE_INICIO_PATH)
        print(f"Plantilla de logo de inicio cargada: {TEMPLATE_INICIO_PATH}")
    else:
        print(f"Aviso: no se encontro '{TEMPLATE_INICIO_PATH}', no habra espera al inicio de partida.")

    region = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE) or FALLBACK_REGION
    print(f"Capturando region inicial: {region}")

    plantilla_x2 = None
    if os.path.exists(TEMPLATE_X2_PATH):
        plantilla_x2 = cv2.imread(TEMPLATE_X2_PATH)
        print(f"Plantilla de elixir x2 cargada: {TEMPLATE_X2_PATH}")
    else:
        print(f"Aviso: no se encontro '{TEMPLATE_X2_PATH}', se jugara siempre como x1.")

    ultimo_x2_timestamp = None   # None = x2 nunca visto   


    frame_count = 0
    prev_time = time.time()
    ultima_jugada_timestamp = 0
    ultimo_logo_timestamp = None   # None = logo nunca visto -> se puede jugar
    torre_tracker = TorreTracker()
    torre_tracker_player = TorreTracker("princess_tower_player")
    vida_tracker = VidaTorresTracker()

    while True:
        if frame_count % REFRESH_WINDOW_EVERY_N_FRAMES == 0:
            nueva = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE)
            if nueva:
                region = nueva

        frame = capturar_frame(sct, region)
        ahora = time.time()

        # --- Rematch: pulsar "Jugar de Nuevo" ---
        if plantilla_rematch is not None and frame_count % REVISAR_REMATCH_CADA_N_FRAMES == 0:
            encontrado, cx, cy, score = buscar_boton_rematch(frame, plantilla_rematch)
            if encontrado:
                x_absoluto = region["left"] + cx
                y_absoluto = region["top"] + cy
                print(f"[rematch] Boton detectado (score={score:.2f}), clic en ({x_absoluto}, {y_absoluto})")
                pyautogui.click(x_absoluto, y_absoluto)
                time.sleep(2)  # da tiempo a que cargue la siguiente partida
                frame_count += 1
                continue

        # --- Logo de inicio: bloquea las jugadas 2 s tras verse ---
        if plantilla_inicio is not None and frame_count % REVISAR_INICIO_CADA_N_FRAMES == 0:
            visto, _, _, score_inicio = buscar_boton_rematch(frame, plantilla_inicio, UMBRAL_MATCH_INICIO)
            if visto:
                if ultimo_logo_timestamp is None or ahora - ultimo_logo_timestamp > 5:
                    torre_tracker = TorreTracker()   # partida nueva: reinicia torres
                    torre_tracker_player = TorreTracker("princess_tower_player")
                    vida_tracker = VidaTorresTracker()   # <-- añadir
                    print(f"[inicio] Logo detectado (score={score_inicio:.2f})")
                ultimo_logo_timestamp = ahora

        # --- Elixir x2: detecta el simbolo ---
        if plantilla_x2 is not None and frame_count % REVISAR_X2_CADA_N_FRAMES == 0:
            visto_x2, _, _, score_x2 = buscar_boton_rematch(frame, plantilla_x2, UMBRAL_MATCH_X2)
            if visto_x2:
                if ultimo_x2_timestamp is None or ahora - ultimo_x2_timestamp > SEGUNDOS_PERSISTENCIA_X2:
                    print(f"[elixir] x2 detectado (score={score_x2:.2f})")
                ultimo_x2_timestamp = ahora

        en_x2 = (
            ultimo_x2_timestamp is not None
            and ahora - ultimo_x2_timestamp <= SEGUNDOS_PERSISTENCIA_X2
        )       
        multiplicador = 1.0 if en_x2 else MULTIPLICADOR_ELIXIR_X1

        # --- Percepcion ---
        estado, resultados_yolo = construir_game_state(frame, modelo_yolo, hud_config, plantillas)
        torre_tracker.actualizar(estado.tropas)

         # --- Percepcion ---
        estado, resultados_yolo = construir_game_state(frame, modelo_yolo, hud_config, plantillas)
        torre_tracker.actualizar(estado.tropas)
        torre_tracker_player.actualizar(estado.tropas)
        estado.vida_torres = vida_tracker.actualizar(frame, estado.tropas, torre_tracker.destruida, torre_tracker_player.destruida)
        estado.torres_destruidas = torre_tracker.destruida

        if frame_count % 30 == 0:
            depurar_barras(frame, estado.tropas)   # quitar al terminar de calibrar

        
        # --- Decision y accion ---
        partida_lista = (
            ultimo_logo_timestamp is None
            or ahora - ultimo_logo_timestamp >= COOLDOWN_INICIO_PARTIDA
        )

        if partida_lista and ahora - ultima_jugada_timestamp > COOLDOWN_SEGUNDOS:
            decision = decidir_jugada(estado, multiplicador)

            if decision is not None:
                if decision["carta"] in HECHIZOS:
                    posicion = calcular_posicion_objetivo_hechizo(decision["carta"], estado)
                else:
                    posicion = calcular_posicion_despliegue(estado, decision["carta"], torre_tracker.destruida)

                jugado = jugar_carta(decision["carta"], estado.mano, hud_config, region, posicion)
                if jugado:
                    ultima_jugada_timestamp = ahora

        #print(estado.resumen())
        #print("Vida torres:", estado.vida_torres)
        

        # --- Visualizacion ---
        vista = resultados_yolo[0].plot(labels=False, conf=False)
        cv2.putText(vista, f"Elixir: {estado.elixir}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(vista, f"Mano: {estado.mano}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if not partida_lista:
            cv2.putText(vista, "ESPERANDO INICIO", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        cv2.putText(vista, "ELIXIR x2" if en_x2 else f"x1 (cartas x{MULTIPLICADOR_ELIXIR_X1})", (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        for t in estado.tropas:
            if t.tipo in TORRES:
                x1, y1, x2, y2 = region_barra(t, vista.shape[1], vista.shape[0])
                cv2.rectangle(vista, (x1, y1), (x2, y2), (255, 255, 0), 1)
                vida = estado.vida_torres.get(clave_torre(t))
                texto = f"{vida:.2f}" if vida is not None else "100%"
                cv2.putText(vista, texto, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(vista, f"FPS: {fps:.1f}", (10, vista.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(PREVIEW_WINDOW_TITLE, vista)
        crono.informar()
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
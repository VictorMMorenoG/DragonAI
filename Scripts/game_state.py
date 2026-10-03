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
MODEL_PATH = r"C:\DragonAI\runs\detect\train-7\weights\best.pt"
HUD_CONFIG_PATH = "hud_config.json"
TEMPLATES_DIR = "templates"

CONFIDENCE_THRESHOLD_YOLO = 0.5
UMBRAL_MATCH_CARTAS = 0.3
ELIXIR_COLOR_BGR = np.array([200, 50, 200])
ELIXIR_TOLERANCIA = 45
MITAD_CAMPO_Y = 0.46
UMBRAL_CERCA_TORRE_Y = 0.53

COOLDOWN_SEGUNDOS = 3


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

SWARMS_ENEMY = {"minions_enemy", "archers_enemy", "goblins_enemy", "spear_goblins_enemy", "squeleton_dragon_enemy"}

TROPAS_OBJETIVO_FIREBALL = {"archers_enemy", "musketeer_enemy", "barbarian_enemy", "squeleton_dragon_enemy"}

AIR_COUNTERS_ENEMY = {"archers_enemy", "musketeer_enemy", "wizard_enemy"}

TANQUES_ENEMY = {"giant_enemy", "battle_ram_enemy"}

MINITANQUES_ENEMY = {"knight_enemy", "minipekka_enemy", "goblin_brawler_enemy", "valkyrie_enemy", "megaminion_enemy"}

TANQUES = {"giant_player"}

MINITANQUES = {"knight_player", "minipekka_player"}

AIR_COUNTERS = {"archers_player", "musketeer_player"}

HECHIZOS = {"arrows", "fireball"}

RADIO_AGRUPACION_HECHIZO = {
    "arrows": 0.15,
    "fireball": 0.10,
}

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
        valor += 0.2

    if estado.elixir > 8:
        valor += 0.2

    if estado.tropas_aliadas:
        valor += 0.2

    # +0.2 si hay mosqueteras o arqueras aliadas en mi lado de la arena
    aircounters_aliados = any(
        t.tipo in AIR_COUNTERS and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_aliadas
    )
    if aircounters_aliados:
        valor += 0.2
    
    return round(min(valor, 1.0), 2)

def val_knight(estado: GameState):
    valor = 0.1

    if estado.tropas_enemigas:
        valor += 0.2

    if estado.tropas_aliadas:
        valor += 0.2

    # +0.2 si hay un tanque enemigo en mi lado de la arena
    tanques_en_mi_lado = any(
        t.tipo in (TANQUES_ENEMY | MINITANQUES_ENEMY) and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if tanques_en_mi_lado:
        valor += 0.2
    
    # +0.2 si hay un antiaereo enemigo en mi lado de la arena
    aircounters_en_mi_lado = any(
        t.tipo in (AIR_COUNTERS) and t.y_norm > MITAD_CAMPO_Y-0.1
        for t in estado.tropas_enemigas
    )
    if aircounters_en_mi_lado:
        valor += 0.2

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
        t.tipo in (TANQUES | MINITANQUES) and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_aliadas
    )
    if tanques_aliados:
        valor += 0.2
    
    return round(min(valor, 1.0), 2)

def val_arrows(estado: GameState):
    valor = 0.0

    tropas_enemigas = [t for t in estado.tropas_enemigas if t.y_norm > 0.3]

    # +0.4 si hay mas de 3 tropas enemigas agrupadas (cerca unas de otras) y hay cartas que mueren a flechas
    cantidad, _ = encontrar_grupo_mas_numeroso(tropas_enemigas, radio=0.15)
    if cantidad > 3 and any(t.tipo in SWARMS_ENEMY for t in tropas_enemigas):
        valor += 0.4

    return round(min(valor, 1.0), 2)

def val_fireball(estado: GameState):
    valor = 0.0

    tropas_enemigas = [t for t in estado.tropas_enemigas if t.y_norm > 0.3]

    # +0.4 si hay mas de 2 tropas enemigas agrupadas (cerca unas de otras) y hay tropas enemigas que mueren a fireball
    cantidad, _ = encontrar_grupo_mas_numeroso(tropas_enemigas, radio=0.1)
    if cantidad > 1 and any(t.tipo in TROPAS_OBJETIVO_FIREBALL for t in tropas_enemigas):
        valor += 0.4

    return round(min(valor, 1.0), 2)

def val_minipekka(estado: GameState):
    valor = 0.1

    # +0.5 si hay un giant enemigo en mi lado de la arena
    giant_en_mi_lado = any(
        t.tipo in TANQUES_ENEMY and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if giant_en_mi_lado:
        valor += 0.5

    # +0.2 si hay mosqueteras o arqueras enemigos en mi lado de la arena
    aircounters_en_mi_lado = any(
        t.tipo in AIR_COUNTERS_ENEMY and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if aircounters_en_mi_lado:
        valor += 0.2

    # +0.3 si hay minitanques enemigos en mi lado de la arena
    minitanques_en_mi_lado = any(
        t.tipo in MINITANQUES_ENEMY and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if minitanques_en_mi_lado:
        valor += 0.3

    # -0.2 si hay swarms enemigos en mi lado de la arena
    swarms_en_mi_lado = any(
        t.tipo in SWARMS_ENEMY and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if swarms_en_mi_lado:
        valor -= 0.2
    
    return round(min(valor, 1.0), 2)

def val_minions(estado: GameState):
    valor = 0.1

    # +0.3 si hay un tanque enemigo en mi lado de la arena
    tanques_en_mi_lado = any(
        t.tipo in (TANQUES_ENEMY | MINITANQUES_ENEMY) and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_enemigas
    )
    if tanques_en_mi_lado:
        valor += 0.3

    if any(t.tipo in AIR_COUNTERS_ENEMY for t in estado.tropas_enemigas):
        valor -= 0.2
    
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
        t.tipo in (TANQUES | MINITANQUES) and t.y_norm > MITAD_CAMPO_Y
        for t in estado.tropas_aliadas
    )
    if tanques_aliados:
        valor += 0.2

    return round(min(valor, 1.0), 2)

def valor_esperar(estado: GameState):
    valor = 0.3

    if estado.elixir > 8:
        valor = 0.1

    return valor

def calcular_posicion_objetivo_hechizo(nombre_carta: str, estado: GameState):
    """Devuelve la posicion (x_norm, y_norm) donde apuntar un hechizo:
    el centro del grupo enemigo mas numeroso. Si no hay enemigos, apunta
    a la torre enemiga mas debil visible (fallback razonable)."""
    tropas_enemigas = estado.tropas_enemigas
    radio = RADIO_AGRUPACION_HECHIZO.get(nombre_carta, 0.12)

    _, centro = encontrar_grupo_mas_numeroso(tropas_enemigas, radio)
    if centro is not None:
        x_centro, y_centro = centro
        y_ajustado = max(0.0, min(1.0, y_centro + 0.05))  # evita salirse de 0.0-1.0
        return (x_centro, y_ajustado)

    # Fallback: si no hay tropas enemigas visibles, apunta a una torre enemiga
    torres_enemigas = [t for t in estado.tropas if t.tipo in TORRES and t.tipo.endswith("_enemy")]
    if torres_enemigas:
        torre = torres_enemigas[0]
        return (torre.x_norm, torre.y_norm)

    return (0.5, 0.3)  # ultimo recurso: centro del lado enemigo

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


def elegir_zona(estado: GameState, nombre_carta: str = None) -> str:
    """Decide dinamicamente que zona usar, mirando si hay amenazas
    enemigas invadiendo mi lado del campo y que tan cerca estan."""
    tropas_enemigas_en_mi_lado = [
        t for t in estado.tropas_enemigas if t.y_norm > 0.45
    ]
    soportes_atras = any(
            t.tipo in (AIR_COUNTERS | MINITANQUES) and t.y_norm > 0.51
            for t in estado.tropas_aliadas
        )
    cartas_atacando = any(
            t.tipo in (AIR_COUNTERS | MINITANQUES | TANQUES) and t.y_norm < 0.5
            for t in estado.tropas_aliadas
        )
    if not tropas_enemigas_en_mi_lado and cartas_atacando:
        return "puente" 
    elif nombre_carta in TANQUES | MINITANQUES and soportes_atras:
        return "puente"
    elif not tropas_enemigas_en_mi_lado:
        return "detras_torre"
    else:
        return "frente_torre"

    

def elegir_carril(estado: GameState) -> float:
    """Devuelve la coordenada x_norm del carril, priorizando el lado
    donde este la amenaza mas avanzada; si no hay amenazas, usa un
    carril por defecto."""
    tropas_enemigas_en_mi_lado = [
        t for t in estado.tropas_enemigas if t.y_norm > MITAD_CAMPO_Y
    ]

    if not tropas_enemigas_en_mi_lado:
        return 0.25  # carril izquierdo por defecto al atacar

    amenaza_mas_cercana = max(tropas_enemigas_en_mi_lado, key=lambda t: t.y_norm)
    return amenaza_mas_cercana.x_norm  # despliega justo en el carril de la amenaza


def calcular_posicion_despliegue(estado: GameState, nombre_carta: str = None):
    zona = elegir_zona(estado, nombre_carta)
    y_norm = ZONAS_Y[zona]
    x_norm = elegir_carril(estado)
    return (x_norm, y_norm)


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
    region = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE) or FALLBACK_REGION
    print(f"Capturando region inicial: {region}")

    frame_count = 0
    prev_time = time.time()
    ultima_jugada_timestamp = 0

    while True:
        if frame_count % REFRESH_WINDOW_EVERY_N_FRAMES == 0:
            nueva = find_window_region(WINDOW_TITLE, exclude_substring=PREVIEW_WINDOW_TITLE)
            if nueva:
                region = nueva

        frame = capturar_frame(sct, region)
        estado, resultados_yolo = construir_game_state(frame, modelo_yolo, hud_config, plantillas)

        ahora = time.time()
        decision = None  # evita usar un valor obsoleto de la iteracion anterior
        print(f"[debug] tiempo desde ultima jugada: {ahora - ultima_jugada_timestamp:.2f}s")
        if ahora - ultima_jugada_timestamp > COOLDOWN_SEGUNDOS:
            decision = decidir_jugada(estado)

            if decision is not None:
                if decision["carta"] in HECHIZOS:
                    posicion = calcular_posicion_objetivo_hechizo(decision["carta"], estado)
                else:
                    posicion = calcular_posicion_despliegue(estado, decision["carta"])

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
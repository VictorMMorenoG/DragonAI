"""
accion.py
---------
Traduce una decision ("jugar tal carta en tal posicion") en clics de raton
reales sobre la ventana del juego, usando pyautogui.

REQUISITOS:
    pip install pyautogui

IMPORTANTE:
    pyautogui trabaja con coordenadas ABSOLUTAS de la pantalla (todo tu
    monitor), no relativas a la ventana del juego. Por eso hay que sumar
    siempre region["left"]/region["top"] a las coordenadas calculadas
    dentro de la ventana.
"""

import time

import pyautogui

# Pequeña pausa de seguridad que pyautogui espera entre acciones.
# 0.1-0.2s suele ser suficiente y no se nota como "lento" jugando.
pyautogui.PAUSE = 0.15

# Mueve el raton a una esquina de la pantalla para abortar si algo va mal
# (poner el raton en la esquina superior izquierda lanza una excepcion).
pyautogui.FAILSAFE = True


def region_relativa_a_pixeles_absolutos(region_rel, region_ventana):
    """Convierte una region relativa (0.0-1.0) del HUD, o un punto
    (x_norm, y_norm), a coordenadas absolutas de pantalla."""
    left, top = region_ventana["left"], region_ventana["top"]
    width, height = region_ventana["width"], region_ventana["height"]

    if len(region_rel) == 4:
        x1p, y1p, x2p, y2p = region_rel
        return (
            left + int(x1p * width),
            top + int(y1p * height),
            left + int(x2p * width),
            top + int(y2p * height),
        )
    else:
        x_norm, y_norm = region_rel
        return left + int(x_norm * width), top + int(y_norm * height)


def encontrar_slot_carta(nombre_carta, mano):
    """Devuelve el indice (0-3) del slot donde esta 'nombre_carta' en la
    mano actual, o None si no esta."""
    for i, carta in enumerate(mano):
        if carta == nombre_carta:
            return i
    return None


def jugar_carta(nombre_carta, mano, hud_config, region_ventana, posicion_objetivo_norm):
    """
    Ejecuta los 2 clics para jugar una carta:
      1. Clic en el slot donde esta la carta en la mano
      2. Clic en la posicion de la arena donde se despliega

    posicion_objetivo_norm: tupla (x_norm, y_norm) relativa a la ventana
    capturada, indicando donde hacer el segundo clic.

    Devuelve True si se pudo ejecutar, False si la carta no estaba en mano.
    """
    slot_idx = encontrar_slot_carta(nombre_carta, mano)
    if slot_idx is None:
        print(f"[accion] '{nombre_carta}' no esta en la mano actual, no se juega.")
        return False

    slot_key = f"carta_slot_{slot_idx + 1}"
    x1, y1, x2, y2 = region_relativa_a_pixeles_absolutos(tuple(hud_config[slot_key]), region_ventana)
    centro_slot = ((x1 + x2) // 2, (y1 + y2) // 2)

    x_destino, y_destino = region_relativa_a_pixeles_absolutos(posicion_objetivo_norm, region_ventana)

    print(f"[accion] Jugando '{nombre_carta}': clic en slot {slot_idx + 1} -> clic en ({x_destino}, {y_destino})")

    pyautogui.click(centro_slot[0], centro_slot[1])
    time.sleep(0.1)
    pyautogui.click(x_destino, y_destino)

    return True

"""
Valida que los indices de clase en los .txt de labels coincidan con lo
esperado geometricamente, usando las torres como ancla (su posicion en
pantalla es fija y conocida, a diferencia de las tropas que se mueven).

Uso:
    python validar_indices.py
    python validar_indices.py --labels dataset/labels/train
"""

import argparse
import sys
from pathlib import Path

import yaml

# Umbrales de posicion normalizada (0.0-1.0) para distinguir "arriba" (enemigo)
# de "abajo" (jugador). Ajusta si tu HUD/captura tiene proporciones distintas.
Y_ENEMY_MAX = 0.45   # y menor que esto -> mitad superior del mapa (rival)
Y_PLAYER_MIN = 0.55  # y mayor que esto -> mitad inferior del mapa (tuya)

# Pares de clases cuya geometria conocemos: (nombre_esperado_arriba, nombre_esperado_abajo)
TOWER_PAIRS = [
    ("king_tower_enemy", "king_tower_player"),
    ("princess_tower_enemy", "princess_tower_player"),
]


def cargar_names(data_yaml_path: Path) -> dict[int, str]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["names"].items()}


def revisar_carpeta(labels_dir: Path, names: dict[int, str]) -> None:
    name_to_idx = {v: k for k, v in names.items()}

    txts = sorted(labels_dir.glob("*.txt"))
    if not txts:
        print(f"  (sin archivos .txt en {labels_dir})")
        return

    errores = 0
    revisadas = 0

    for txt in txts:
        lines = [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            cls_idx = int(parts[0])
            y_center = float(parts[2])

            cls_name = names.get(cls_idx)
            if cls_name is None:
                print(f"  [!] {txt.name}: indice {cls_idx} no existe en data.yaml")
                errores += 1
                continue

            for enemy_name, player_name in TOWER_PAIRS:
                if cls_name == enemy_name and y_center > Y_PLAYER_MIN:
                    print(
                        f"  [!] {txt.name}: '{enemy_name}' (idx {cls_idx}) "
                        f"aparece abajo (y={y_center:.2f}) -> probablemente deberia "
                        f"ser '{player_name}' (idx {name_to_idx.get(player_name)})"
                    )
                    errores += 1
                    revisadas += 1
                elif cls_name == player_name and y_center < Y_ENEMY_MAX:
                    print(
                        f"  [!] {txt.name}: '{player_name}' (idx {cls_idx}) "
                        f"aparece arriba (y={y_center:.2f}) -> probablemente deberia "
                        f"ser '{enemy_name}' (idx {name_to_idx.get(enemy_name)})"
                    )
                    errores += 1
                    revisadas += 1
                elif cls_name in (enemy_name, player_name):
                    revisadas += 1

    print(f"  Instancias de torres revisadas: {revisadas}")
    print(f"  Inconsistencias encontradas:    {errores}")
    if errores == 0 and revisadas > 0:
        print("  -> Los indices de torres cuadran con la geometria esperada.")
    elif revisadas == 0:
        print("  -> No se encontraron instancias de torres en esta carpeta (revisa el path).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/data.yaml", help="Ruta a data.yaml")
    parser.add_argument("--labels", default=None, help="Carpeta concreta de labels a revisar")
    args = parser.parse_args()

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        print(f"No encuentro {data_yaml_path}")
        sys.exit(1)

    names = cargar_names(data_yaml_path)

    if args.labels:
        carpetas = [Path(args.labels)]
    else:
        base = data_yaml_path.parent
        carpetas = [base / "labels" / "train", base / "labels" / "val"]

    for carpeta in carpetas:
        print(f"\n=== Revisando {carpeta} ===")
        if not carpeta.exists():
            print("  (carpeta no encontrada, saltando)")
            continue
        revisar_carpeta(carpeta, names)


if __name__ == "__main__":
    main()

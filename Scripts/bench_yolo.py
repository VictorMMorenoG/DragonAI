import time
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"C:\DragonAI\runs\detect\train-10\weights\best.pt"
modelo = YOLO(MODEL_PATH)
frame = np.zeros((1041, 714, 3), dtype=np.uint8)   # tamano parecido a tu ventana

for kwargs in ({"device": 0}, {"device": 0, "half": True}, {"device": 0, "imgsz": 640}):
    try:
        for _ in range(3):  # calentamiento
            modelo.predict(frame, verbose=False, **kwargs)
        t = time.perf_counter()
        for _ in range(30):
            modelo.predict(frame, verbose=False, **kwargs)
        ms = (time.perf_counter() - t) / 30 * 1000
        print(f"{kwargs}: {ms:.1f} ms/frame ({1000 / ms:.0f} fps)")
    except Exception as e:
        print(f"{kwargs}: ERROR -> {e}")
"""Modo 6-7: la camara solo detecta el gesto del meme y reproduce el gif.

Sin recuadro, sin filtros y sin guardar fotos: es la version enfocada solo en
el 6-7. Mueve las dos manos como una balanza (una arriba, otra abajo, alternando)
y salta el gif de tenor.gif.

Reutiliza el detector, el lector de gif y el gesto de los otros modulos:
  - crear_detector, cargar_gif, ReproductorGif, cartel_67  -> main.py
  - Gesto67                                                 -> gesto67.py

Ejecutar:  python modo67.py     (ESC para salir)
"""

import os

import cv2
import mediapipe as mp

from main import (
    CAMARA,
    GIF_67,
    GIF_ALTO,
    ReproductorGif,
    cargar_gif,
    cartel_67,
    crear_detector,
    dibujar_diag,
)
from gesto67 import Gesto67
import sonido

VENTANA = "PhotoMake - Modo 6 7"


def main():
    print("\n=== PhotoMake - Modo 6 7 ===\n")
    print("Mueve las dos manos como una balanza (una arriba, otra abajo).")
    print("ESC para salir.\n")

    detector = crear_detector()
    detector67 = Gesto67()

    gif67 = ReproductorGif([])
    if os.path.exists(GIF_67):
        try:
            gif67 = ReproductorGif(cargar_gif(GIF_67, GIF_ALTO))
        except Exception as e:
            print(f"No se pudo cargar {GIF_67}: {e}")
    else:
        print(f"Aviso: no encuentro {GIF_67}, el 6-7 saldra sin gif.")

    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara {CAMARA}.")
        return

    # detect_for_video exige timestamps crecientes: el contador de frames los da.
    nframe = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # espejo, para apuntar de forma natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            nframe += 1
            ms = nframe * 33
            resultado = detector.detect_for_video(imagen_mp, ms)

            if detector67.actualizar(resultado.hand_landmarks, ms):
                print("6 7")
                gif67.disparar()
                sonido.reproducir_67()

            # El gif manda mientras suena; si no se cargo, sale el cartel de texto.
            if gif67.dibujar(frame):
                pass
            elif detector67.celebrando(ms) and not gif67.frames:
                cartel_67(frame)

            # Lector de diagnostico, para ver que esta midiendo el detector.
            dibujar_diag(frame, detector67.diag)

            cv2.imshow(VENTANA, frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

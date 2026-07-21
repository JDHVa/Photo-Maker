"""Ventana magica: el filtro se aplica solo dentro del recuadro de tus manos.

Haz una "L" con cada mano (pulgar e indice), como cuando encuadras una foto: en
ese recuadro se ve el filtro y el resto de la imagen queda normal. Las teclas
1-6 cambian de filtro en vivo, 0 lo apaga.

El recuadro se calcula como la caja que contiene las puntas de pulgar e indice
de las manos detectadas. Asi funciona igual con una mano (pinza) que con dos
(encuadre clasico), sin tener que distinguir gestos.
"""

import os
import sys
from datetime import datetime

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "filtros"))

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import registro

MODELO = os.path.join(RAIZ, "modelos", "hand_landmarker.task")
OUTPUT_DIR = os.path.join(RAIZ, "output")
CAMARA = 2

# Indices de landmarks de MediaPipe
PULGAR, INDICE = 4, 8

SUAVIZADO = 0.4    # 0 = congelado, 1 = sin suavizar (tiembla)
MARGEN = 0.06      # holgura alrededor de los dedos, en fraccion del recuadro
MIN_LADO = 60      # px minimos por lado para considerar el encuadre utilizable

# Los filtros guardan estado entre frames (la semilla del kmeans, y las capas
# de papel/trama cacheadas por resolucion) y asumen que el tamaño no cambia.
# El recuadro de las manos cambia en cada frame, asi que el ROI se normaliza
# siempre a este tamaño antes de filtrar. Sin esto el cache crece una entrada
# por frame (fuga de memoria) y la paleta vuelve a parpadear.
PROCESO = (384, 288)


def crear_detector():
    opciones = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODELO),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opciones)


def puntos_de_encuadre(resultado, w, h):
    """Puntas de pulgar e indice de cada mano, en pixeles."""
    puntos = []
    for mano in resultado.hand_landmarks:
        for idx in (PULGAR, INDICE):
            lm = mano[idx]
            puntos.append((int(lm.x * w), int(lm.y * h)))
    return puntos


def recuadro_de(puntos, w, h):
    """Caja que contiene los puntos, con un margen y recortada a la imagen."""
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    mx = int((x2 - x1) * MARGEN)
    my = int((y2 - y1) * MARGEN)
    return (
        max(0, x1 - mx), max(0, y1 - my),
        min(w - 1, x2 + mx), min(h - 1, y2 + my),
    )


def aplicar_en_recuadro(img, caja, filtro):
    """Aplica el filtro solo dentro de la caja, con corte duro en el borde.

    El ROI se estira al tamaño de proceso fijo y se devuelve al tamaño real.
    Nota para Pixel Art: su rejilla se calcula sobre el tamaño de proceso, asi
    que si el recuadro tiene otra proporcion los pixelotes salen un poco
    rectangulares. Es el precio de que el cache no se fugue.
    """
    x1, y1, x2, y2 = caja
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img

    chico = cv2.resize(roi, PROCESO, interpolation=cv2.INTER_AREA)
    filtrado = filtro(chico)
    img[y1:y2, x1:x2] = cv2.resize(
        filtrado, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR
    )
    return img


def dibujar_visor(img, caja, activo=True):
    """Recuadro estilo visor de camara: esquinas marcadas y guias tenues."""
    x1, y1, x2, y2 = caja
    color = (80, 255, 80) if activo else (150, 150, 150)
    largo = max(18, int(min(x2 - x1, y2 - y1) * 0.18))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    for cx, dx in ((x1, 1), (x2, -1)):
        for cy, dy in ((y1, 1), (y2, -1)):
            cv2.line(img, (cx, cy), (cx + dx * largo, cy), color, 3)
            cv2.line(img, (cx, cy), (cx, cy + dy * largo), color, 3)

    cv2.putText(img, f"{x2 - x1}x{y2 - y1}", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img


def barra_estado(img, nombre_filtro, pista):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 46), (w, h), (0, 0, 0), -1)
    cv2.putText(img, f"[{nombre_filtro}]", (12, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 255, 80), 2, cv2.LINE_AA)
    cv2.putText(img, pista, (12, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(img, "1-6 filtro  0 ninguno  ESPACIO guardar  ESC salir",
                (w - 430, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (150, 150, 150), 1, cv2.LINE_AA)
    return img


def main():
    if not os.path.exists(MODELO):
        print(f"Falta el modelo en {MODELO}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== PhotoMake - Ventana magica ===\n")
    print("Haz una 'L' con cada mano: el filtro se ve solo en ese recuadro.")
    for tecla, nombre, _, _, desc in registro.FILTROS:
        print(f"  [{tecla}] {nombre:<10} {desc}")
    print("  [0] ninguno")
    print("\nESPACIO: guardar | ESC: salir\n")

    print("Cargando filtros...")
    filtros = registro.cargar_todos()
    detector = crear_detector()

    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara {CAMARA}.")
        return

    activo = "2"          # empieza en Comic
    caja_suave = None
    # detect_for_video exige timestamps estrictamente crecientes y revienta si
    # se repite uno. Un contador de frames lo garantiza; usar el reloj no, dos
    # frames pueden caer en el mismo milisegundo.
    nframe = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Espejo: sin esto mover la mano a la derecha la mueve a la izquierda
            # en pantalla y es imposible apuntar. Va antes de detectar para que
            # las coordenadas coincidan con lo que se dibuja.
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            nframe += 1
            resultado = detector.detect_for_video(imagen_mp, nframe * 33)

            puntos = puntos_de_encuadre(resultado, w, h)
            nombre_filtro = filtros[activo][0] if activo in filtros else "ninguno"

            if puntos:
                caja = np.array(recuadro_de(puntos, w, h), np.float32)
                # Suavizado exponencial: los landmarks tiemblan entre frames y
                # sin esto el recuadro vibra aunque tengas la mano quieta.
                caja_suave = caja if caja_suave is None else (
                    SUAVIZADO * caja + (1 - SUAVIZADO) * caja_suave
                )
                x1, y1, x2, y2 = caja_suave.astype(int)

                # Con ciertos gestos (un pulgar arriba, por ejemplo) las dos
                # puntas quedan casi alineadas y el recuadro sale degenerado,
                # de unos pocos pixeles. No sirve para filtrar ni recortar.
                listo = (x2 - x1) >= MIN_LADO and (y2 - y1) >= MIN_LADO

                if listo and activo in filtros:
                    aplicar_en_recuadro(frame, (x1, y1, x2, y2), filtros[activo][1])

                # Copia sin anotaciones: el recorte que se guarda sale de aqui,
                # ya con el filtro aplicado pero sin el visor ni los puntos.
                limpio = frame.copy()

                for p in puntos:
                    cv2.circle(frame, p, 6, (80, 255, 80), -1)
                dibujar_visor(frame, (x1, y1, x2, y2), activo=listo)

                if not listo:
                    pista = "abre mas las manos"
                elif len(puntos) >= 4:
                    pista = "listo - ESPACIO para guardar"
                else:
                    pista = "1 mano (usa las dos para encuadrar mejor)"
            else:
                caja_suave = None
                listo = False
                limpio = frame
                pista = "muestra las manos a la camara"

            barra_estado(frame, nombre_filtro, pista)
            cv2.imshow("PhotoMake - Ventana magica", frame)

            tecla = cv2.waitKey(1) & 0xFF
            caracter = chr(tecla) if 32 <= tecla < 127 else ""

            if tecla == 27:
                break
            if caracter == "0":
                activo = "0"
                print("Filtro: ninguno")
            elif caracter in filtros:
                activo = caracter
                print(f"Filtro: {filtros[activo][0]}")
            elif tecla == 32:
                if caja_suave is None:
                    print("No veo tus manos.")
                elif not listo:
                    print("Encuadre demasiado chico, abre mas las manos.")
                else:
                    x1, y1, x2, y2 = caja_suave.astype(int)
                    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ruta = os.path.join(OUTPUT_DIR, f"Ventana_{sello}.png")
                    cv2.imwrite(ruta, limpio[y1:y2, x1:x2])
                    print(f"Guardado en: {ruta}  ({x2 - x1}x{y2 - y1})")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

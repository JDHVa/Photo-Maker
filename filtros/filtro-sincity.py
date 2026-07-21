"""Sin City: blanco y negro extremo con los rojos intactos.

El detalle dificil no es el blanco y negro, es la mascara de rojos. La piel es
anaranjada y cae peligrosamente cerca del rojo en el circulo cromatico, asi que
filtrar solo por tono deja la cara a color y arruina el efecto. La solucion es
exigir ademas SATURACION alta: la piel rara vez pasa de ~150, mientras que una
playera o unos labios rojos se van por encima de 170.
"""

import cv2
import numpy as np

import comun

# --- Palancas ---
TONO = 8            # ancho del tono rojo aceptado (0-180 en OpenCV)
SAT_MINIMA = 165    # subir si la piel se cuela; bajar si no agarra los rojos
VAL_MINIMO = 60     # ignora rojos casi negros, que solo serian ruido
CONTRASTE = 1.55    # el look Sin City vive de negros aplastados
DUREZA = 30         # mas bajo = curva mas abrupta, menos medios tonos

# Curva en S: aplasta las sombras y quema las luces hasta dejar casi solo
# blancos y negros, que es lo que define el look grafico de Sin City.
_i = np.arange(256, dtype=np.float32)
_LUT_SC = np.clip(
    255 / (1 + np.exp(-CONTRASTE * (_i - 128) / DUREZA)), 0, 255
).astype(np.uint8)


def filtro_sincity(img):
    # 1. Base en blanco y negro con negros aplastados
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.LUT(gray, _LUT_SC)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. Mascara de rojos. El tono rojo esta partido en los dos extremos del
    # rango (0 y 180), asi que hay que capturar las dos puntas.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    baja = cv2.inRange(hsv, (0, SAT_MINIMA, VAL_MINIMO), (TONO, 255, 255))
    alta = cv2.inRange(hsv, (180 - TONO, SAT_MINIMA, VAL_MINIMO), (180, 255, 255))
    mascara = cv2.bitwise_or(baja, alta)

    # Limpiar puntitos sueltos y suavizar el borde para que no quede aserrado
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mascara = cv2.GaussianBlur(mascara, (5, 5), 0)

    # 3. Rojos mas intensos de lo normal, para que resalten contra el gris
    rojo = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rojo[:, :, 1] = np.clip(rojo[:, :, 1] * 1.35, 0, 255).astype(np.uint8)
    rojo = cv2.cvtColor(rojo, cv2.COLOR_HSV2BGR)

    # 4. Mezclar segun la mascara
    m = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    return (rojo * m + base * (1 - m)).astype(np.uint8)


if __name__ == "__main__":
    comun.correr("Sin City", filtro_sincity)

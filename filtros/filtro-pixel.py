"""Pixel art de 8 bits.

Dos detalles hacen la diferencia entre "foto con bloques" y pixel art de verdad:

1. La paleta es FIJA y pequeña (16 colores). Un kmeans daria colores optimos
   para cada frame, pero justamente lo que define el look retro es que la
   consola no podia elegir: siempre eran los mismos 16 colores.
2. El dithering ordenado (matriz de Bayer). Con solo 16 colores los degradados
   se cortan en bandas feas; el dithering las rompe en el patron de puntos
   entrecruzados clasico de la era de 8/16 bits.
"""

import cv2
import numpy as np

import comun

# --- Palancas ---
ANCHO = 160        # resolucion horizontal en "pixelotes"
DITHER = 8          # 0 = bandas duras, 20+ ya se ve moteado y sucio
SATURACION = 1.15   # ver nota en el paso 2. Arriba de 1.2 la piel se va a cafe

# Paleta de PICO-8: 16 colores diseñados para verse bien juntos.
PALETA_RGB = [
    (0x00, 0x00, 0x00), (0x1D, 0x2B, 0x53), (0x7E, 0x25, 0x53), (0x00, 0x87, 0x51),
    (0xAB, 0x52, 0x36), (0x5F, 0x57, 0x4F), (0xC2, 0xC3, 0xC7), (0xFF, 0xF1, 0xE8),
    (0xFF, 0x00, 0x4D), (0xFF, 0xA3, 0x00), (0xFF, 0xEC, 0x27), (0x00, 0xE4, 0x36),
    (0x29, 0xAD, 0xFF), (0x83, 0x76, 0x9C), (0xFF, 0x77, 0xA8), (0xFF, 0xCC, 0xAA),
]

# Rampa de piel. PICO-8 salta de FFCCAA (piel clara) directo a FF77A8 (rosa):
# no tiene medios tonos de piel, asi que una cara sombreada queda con un anillo
# rosa en la transicion. Estos dos tonos rellenan ese hueco hacia AB5236.
# Es una licencia sobre la paleta original, pero esto es un filtro de caras.
PALETA_RGB += [
    (0xE8, 0xA8, 0x7C),
    (0xC7, 0x7B, 0x5A),
]
# OpenCV trabaja en BGR
PALETA = np.array([(b, g, r) for r, g, b in PALETA_RGB], np.float32)

# Matriz de Bayer 4x4: el orden clasico del dithering ordenado
BAYER = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], np.float32) / 16.0 - 0.5

_cache = {}


def filtro_pixel(img):
    h, w = img.shape[:2]
    alto = max(1, int(ANCHO * h / w))

    # 1. Bajar a resolucion de consola. INTER_AREA promedia el bloque completo,
    # que es lo que da bordes limpios; INTER_LINEAR dejaria restos borrosos.
    chico = cv2.resize(img, (ANCHO, alto), interpolation=cv2.INTER_AREA)

    # 2. Saturar ANTES de cuantizar. La paleta no tiene tonos medios: un rojo
    # apagado cae mas cerca del cafe (AB5236) que del rojo vivo (FF004D) y la
    # imagen sale terrosa. Al saturar primero, cada color se compromete con la
    # entrada viva que le corresponde.
    hsv = cv2.cvtColor(chico, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * SATURACION, 0, 255).astype(np.uint8)
    chico = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. Dithering ordenado antes de cuantizar
    if (alto, ANCHO) not in _cache:
        _cache[(alto, ANCHO)] = np.tile(
            BAYER, (alto // 4 + 1, ANCHO // 4 + 1)
        )[:alto, :ANCHO, None] * DITHER
    chico = np.clip(chico.astype(np.float32) + _cache[(alto, ANCHO)], 0, 255)

    # 4. Cada pixel al color mas cercano de la paleta. A esta resolucion son
    # ~17k pixeles x 16 colores, asi que la fuerza bruta vectorizada sobra.
    dist = ((chico[:, :, None, :] - PALETA[None, None, :, :]) ** 2).sum(axis=3)
    indices = dist.argmin(axis=2)
    chico = PALETA.astype(np.uint8)[indices]

    # 5. Volver al tamaño original. INTER_NEAREST es obligatorio: cualquier otra
    # interpolacion suavizaria los bordes y arruinaria el efecto.
    return cv2.resize(chico, (w, h), interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    comun.correr("Pixel Art", filtro_pixel)

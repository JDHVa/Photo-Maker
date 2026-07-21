"""Manga en blanco y negro con screentones.

A diferencia de los filtros de color, aqui NO hay grises: el manga impreso solo
tiene tinta negra y papel blanco. Los "grises" son una ilusion optica creada por
tramas de puntos de distinta densidad (screentone). Por eso el paso clave es
posterizar la luminancia en zonas planas ANTES de aplicar la trama: eso produce
parches de screentone bien definidos como en el papel, en vez de un degradado.
"""

import cv2
import numpy as np

import comun

# --- Palancas ---
FREQ_TRAMA = 0.60        # mas bajo = puntos mas grandes
ZONAS = 4                # niveles de screentone (4 es lo tipico en manga)
CONTRASTE = 1.5          # el manga vive de blancos y negros extremos
LINEAS_VELOCIDAD = True  # lineas radiales en las orillas

_cache = {}


def _construir_capas(h, w):
    trama = comun.trama_halftone(h, w, FREQ_TRAMA)

    # Lineas de velocidad: radiales desde el centro, solo en el anillo exterior
    # para que no tapen la cara.
    lineas = np.full((h, w), 255, np.uint8)
    if LINEAS_VELOCIDAD:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = xx - w / 2, yy - h / 2
        ang = np.arctan2(dy, dx)
        rad = np.sqrt(dx**2 + dy**2) / (min(h, w) / 2)
        franja = np.sin(ang * 54) > 0.88
        # Empiezan bien afuera para que no invadan la cara ni la tapen
        fuera = rad > 1.05
        lineas[franja & fuera] = 0

    return trama, lineas


def filtro_manga(img):
    h, w = img.shape[:2]
    if (h, w) not in _cache:
        _cache[(h, w)] = _construir_capas(h, w)
    trama, lineas = _cache[(h, w)]

    # 1. Luminancia con contraste alto y superficies limpias. El bilateral quita
    # el ruido de piel sin comerse los bordes, que si no ensucia la trama.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    gray = cv2.convertScaleAbs(gray, alpha=CONTRASTE, beta=-40)

    # 2. Posterizar a zonas planas. Este es el paso que hace que se lea como
    # screentone de manga y no como un halftone continuo de foto.
    paso = 256 // ZONAS
    zonas = (gray // paso) * paso + paso // 2

    # 3. Trama: donde la zona supera el patron queda papel, si no queda tinta.
    manga = np.where(zonas > trama, 255, 0).astype(np.uint8)

    # 4. Lineas de tinta encima. cv2.min hace que el negro siempre gane.
    manga = cv2.min(manga, comun.bordes_tinta(img, block=9, c=6))
    manga = cv2.min(manga, lineas)

    return cv2.cvtColor(manga, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    comun.correr("Manga", filtro_manga)

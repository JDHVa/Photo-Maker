import cv2
import numpy as np
import comun

FREQ_TRAMA = 0.60
ZONAS = 4
CONTRASTE = 1.5
LINEAS_VELOCIDAD = True

_cache = {}


def _construir_capas(h, w):
    trama = comun.trama_halftone(h, w, FREQ_TRAMA)

    lineas = np.full((h, w), 255, np.uint8)
    if LINEAS_VELOCIDAD:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = xx - w / 2, yy - h / 2
        ang = np.arctan2(dy, dx)
        rad = np.sqrt(dx**2 + dy**2) / (min(h, w) / 2)
        franja = np.sin(ang * 54) > 0.88
        fuera = rad > 1.05
        lineas[franja & fuera] = 0

    return trama, lineas


def filtro_manga(img):
    h, w = img.shape[:2]
    if (h, w) not in _cache:
        _cache[(h, w)] = _construir_capas(h, w)
    trama, lineas = _cache[(h, w)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    gray = cv2.convertScaleAbs(gray, alpha=CONTRASTE, beta=-40)

    paso = 256 // ZONAS
    zonas = (gray // paso) * paso + paso // 2

    manga = np.where(zonas > trama, 255, 0).astype(np.uint8)

    manga = cv2.min(manga, comun.bordes_tinta(img, block=9, c=6))
    manga = cv2.min(manga, lineas)

    return cv2.cvtColor(manga, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    comun.correr("Manga", filtro_manga)

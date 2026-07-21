"""Pagina de comic impresa: puntos Ben-Day, papel envejecido, tinta roja y marco.

Sobre el filtro suave se apilan las capas que hacen que se lea como IMPRESION y
no como una foto estilizada. Todas son estaticas salvo la trama, asi que se
precalculan una vez por resolucion.
"""

import cv2
import numpy as np

import comun

# --- Palancas para calibrar la intensidad del efecto ---
FREQ_TRAMA = 0.55            # mas bajo = puntos Ben-Day mas grandes
TINTA_ROJA = (30, 45, 165)   # BGR del lavado rojo
LAVADO_ROJO = 0.30           # 0 = sin lavado, 0.42 ya se ve lavado

_shader = comun.CelShader(k=8)
_cache = {}


def _construir_capas(h, w):
    return (
        comun.trama_halftone(h, w, FREQ_TRAMA),
        comun.capa_papel(h, w),
        comun.lut_atenuacion(),
        np.full((h, w, 3), TINTA_ROJA, np.uint8),
    )


def filtro_comic(img):
    h, w = img.shape[:2]
    if (h, w) not in _cache:
        _cache[(h, w)] = _construir_capas(h, w)
    trama, papel, lut, rojo = _cache[(h, w)]

    small = comun.reducir(img)

    # 1. Suavizar regiones de color manteniendo bordes nitidos
    smooth = cv2.pyrMeanShiftFiltering(small, sp=8, sr=30)

    # 2. Colores planos con paleta estable entre frames
    quantized = _shader.cuantizar(smooth)

    # 3. Saturacion intensa + contraste tipo comic
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255).astype(np.uint8)
    quantized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    quantized = cv2.convertScaleAbs(quantized, alpha=1.3, beta=15)

    quantized = comun.ampliar(quantized, w, h)

    # 4. Componer color + bordes de tinta
    edges = cv2.cvtColor(comun.bordes_tinta(img, block=9, c=7), cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(quantized, edges)

    # 5. Puntos Ben-Day, solo en sombras y medios tonos
    result = comun.aplicar_halftone(result, trama, lut)

    # 6. Papel + viñeteado (una sola capa multiply precalculada)
    result = cv2.multiply(result, papel, scale=1 / 255.0)

    # 7. Lavado rojo de tinta. El multiply del paso 6 solo puede restar luz, asi
    # que no alcanza para volver calido un fondo frio: hace falta sumar rojo.
    result = cv2.addWeighted(result, 1 - LAVADO_ROJO, rojo, LAVADO_ROJO, 0)

    # 8. Chromatic aberration sutil
    result = comun.aberracion_cromatica(result, px=1)

    # 9. Marco de viñeta
    return comun.marco_vineta(result)


if __name__ == "__main__":
    comun.correr("Comic", filtro_comic)

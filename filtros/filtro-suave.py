"""Estilo comic suave: colores planos saturados con bordes de tinta."""

import cv2
import numpy as np

import comun

_shader = comun.CelShader(k=8)


def filtro_spiderverse(img):
    h, w = img.shape[:2]

    # El color se procesa a media resolucion (son tonos planos, no se nota al
    # reescalar) y los bordes a resolucion completa para que queden nitidos.
    small = comun.reducir(img)

    # 1. Suavizar regiones de color manteniendo bordes nitidos
    # sp=8 a media resolucion equivale al sp=15 original a resolucion completa
    smooth = cv2.pyrMeanShiftFiltering(small, sp=8, sr=30)

    # 2. Colores planos con paleta estable entre frames
    quantized = _shader.cuantizar(smooth)

    # 3. Saturacion intensa + contraste tipo comic
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255).astype(np.uint8)
    quantized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    quantized = cv2.convertScaleAbs(quantized, alpha=1.3, beta=15)

    # Volver a resolucion completa antes de aplicar las lineas
    quantized = comun.ampliar(quantized, w, h)

    # 4. Componer color + bordes de tinta
    edges = cv2.cvtColor(comun.bordes_tinta(img, block=9, c=7), cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(quantized, edges)

    # 5. Chromatic aberration sutil
    return comun.aberracion_cromatica(result, px=1)


if __name__ == "__main__":
    comun.correr("Suave", filtro_spiderverse)

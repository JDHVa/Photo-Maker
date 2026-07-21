"""Estilo anime: piel muy lisa, paleta pastel, lineas finas y bloom.

Es lo opuesto al filtro comic en casi todo: alli la saturacion sube al doble y
los negros se aplastan; aqui la saturacion BAJA y los negros se levantan, que es
lo que produce el look pastel y aireado del anime moderno.
"""

import cv2
import numpy as np

import comun

_shader = comun.CelShader(k=9)


def filtro_anime(img):
    h, w = img.shape[:2]
    small = comun.reducir(img)

    # 1. Piel ultra-lisa: varias pasadas suaves > una pasada agresiva
    smooth = small
    for _ in range(4):
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)

    # 2. Cel shading: pocos tonos planos, luego suavizar las bandas para que no
    # se vea como posterizado duro
    cel = _shader.cuantizar(smooth)
    cel = cv2.medianBlur(cel, 3)

    # 3. Paleta pastel: bajar saturacion y levantar los negros
    hsv = cv2.cvtColor(cel, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0, 255).astype(np.uint8)
    cel = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cel = cv2.convertScaleAbs(cel, alpha=0.9, beta=25)

    # Tinte calido sutil en los medios tonos
    cel = cel.astype(np.int16)
    cel[:, :, 2] += 8   # rojo
    cel[:, :, 0] -= 5   # azul
    cel = np.clip(cel, 0, 255).astype(np.uint8)

    # 4. Bloom: brillo etereo en las zonas claras
    gray_bloom = cv2.cvtColor(cel, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray_bloom, 200, 255, cv2.THRESH_BINARY)
    bright = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)
    bloom = cv2.GaussianBlur(bright, (21, 21), 0)
    cel = cv2.addWeighted(cel, 0.85, bloom, 0.15, 0)

    cel = comun.ampliar(cel, w, h)

    # 5. Lineas delgadas y delicadas: bloque mas chico y C mas alto que en el
    # comic, y sin dilatar
    edges = cv2.cvtColor(comun.bordes_tinta(img, block=7, c=9), cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(cel, edges)


if __name__ == "__main__":
    comun.correr("Anime", filtro_anime)

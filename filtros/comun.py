"""Utilidades compartidas por los filtros de PhotoMake.

Aqui vive lo que se repetia en todos los filtros: el loop de camara, la
cuantizacion de color estable entre frames, los bordes de tinta y las capas
de impresion (trama Ben-Day, papel, viñeteado).
"""

import cv2
import numpy as np
import os
from datetime import datetime

# Anclado a la raiz del proyecto para que las fotos caigan siempre en el mismo
# lugar, corras el filtro desde donde lo corras.
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
)
CAMARA = 2


# --------------------------------------------------------------------------
# Escalado
# --------------------------------------------------------------------------

def reducir(img, escala=2):
    """Baja la resolucion para procesar color. INTER_AREA promedia, que es lo
    correcto al reducir; INTER_LINEAR dejaria aliasing."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // escala, h // escala), interpolation=cv2.INTER_AREA)


def ampliar(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# Cuantizacion de color
# --------------------------------------------------------------------------

class CelShader:
    """Reduce la imagen a tonos planos con K-Means.

    Guarda los labels del frame anterior y los usa como semilla del siguiente.
    Esto hace dos cosas a la vez: acelera el kmeans (converge en 1 iteracion en
    vez de reintentar desde cero) y mantiene la paleta estable entre frames.
    Sin la semilla, KMEANS_RANDOM_CENTERS elige centros distintos cada frame y
    los colores parpadean en video.
    """

    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    def __init__(self, k=8):
        self.k = k
        self._labels = None

    def cuantizar(self, img):
        data = np.float32(img).reshape((-1, 3))
        if self._labels is not None and len(self._labels) == len(data):
            _, labels, centers = cv2.kmeans(
                data, self.k, self._labels, self.CRITERIA, 1,
                cv2.KMEANS_USE_INITIAL_LABELS,
            )
        else:
            _, labels, centers = cv2.kmeans(
                data, self.k, None, self.CRITERIA, 1, cv2.KMEANS_RANDOM_CENTERS,
            )
        self._labels = labels
        return np.uint8(centers)[labels.flatten()].reshape(img.shape)


# --------------------------------------------------------------------------
# Tinta
# --------------------------------------------------------------------------

def bordes_tinta(img, block=9, c=7, blur=5):
    """Mascara de bordes tipo tinta. Devuelve 0 en la linea, 255 en el resto,
    lista para combinar con bitwise_and."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, blur)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c
    )


def aberracion_cromatica(img, px=1):
    """Desplaza los canales rojo y azul en direcciones opuestas."""
    b, g, r = cv2.split(img)
    return cv2.merge([np.roll(b, -px, axis=1), g, np.roll(r, px, axis=1)])


# --------------------------------------------------------------------------
# Capas de impresion
# --------------------------------------------------------------------------

def trama_halftone(h, w, freq=0.55):
    """Rejilla de puntos Ben-Day rotada 45 grados.

    En impresion real los puntos del canal negro van a 45 grados; ese angulo es
    lo que hace que se lea como impresion y no como una cuadricula de pixeles.
    El producto sin(u)*sin(v) sobre coordenadas rotadas da esa rejilla sin
    recorrer pixel por pixel.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + yy) * freq
    v = (xx - yy) * freq
    return (((np.sin(u) * np.sin(v)) + 1) / 2 * 255).astype(np.uint8)


def lut_atenuacion(umbral=190, fuerza=0.45):
    """Cuanto oscurecer el hueco entre puntos segun la luminancia.

    Se desvanece a 0 por encima del umbral para que las luces queden limpias:
    si los puntos entran en las zonas claras, la cara se ve sucia.
    """
    i = np.arange(256, dtype=np.float32)
    peso = np.clip((umbral - i) * (fuerza / umbral), 0, fuerza)
    return np.clip((1 - peso) * 255, 0, 255).astype(np.uint8)


def aplicar_halftone(img, trama, lut):
    """Aplica la trama de puntos. Todo en enteros con ops nativas de OpenCV:
    5x mas rapido que hacer la aritmetica en float sobre cada pixel."""
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    factor = cv2.max(cv2.compare(lum, trama, cv2.CMP_GT), cv2.LUT(lum, lut))
    return cv2.multiply(img, cv2.cvtColor(factor, cv2.COLOR_GRAY2BGR), scale=1 / 255.0)


def capa_papel(h, w, tinte=(0.62, 0.80, 1.00), vigor=0.50, semilla=0):
    """Papel envejecido con viñeteado, listo para mezclar en multiply.

    `tinte` es un factor BGR: como el multiply solo puede restar luz, el tono
    calido se logra apagando el azul y el verde, no subiendo el rojo.
    """
    papel = np.random.default_rng(semilla).normal(238, 8, (h, w)).clip(200, 255)
    papel = cv2.GaussianBlur(papel.astype(np.float32), (3, 3), 0)
    papel = cv2.merge([papel, papel, papel])
    for canal, factor in enumerate(tinte):
        papel[:, :, canal] *= factor

    kx = cv2.getGaussianKernel(w, w * 0.45)
    ky = cv2.getGaussianKernel(h, h * 0.45)
    vig = ky @ kx.T
    vig = (vig / vig.max()) * vigor + (1 - vigor)
    papel *= vig[:, :, None].astype(np.float32)
    return np.clip(papel, 0, 255).astype(np.uint8)


def marco_vineta(img, color=(20, 20, 20)):
    """Marco con margen interior blanco, como el borde de una viñeta."""
    h, w = img.shape[:2]
    m = max(6, int(min(h, w) * 0.025))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), m)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, max(2, m // 3))
    return img


# --------------------------------------------------------------------------
# Loop de camara
# --------------------------------------------------------------------------

def correr(nombre, filtro, camara=CAMARA):
    """Abre la camara, aplica `filtro` a cada frame y guarda con ESPACIO."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n=== PhotoMake - {nombre} (Video en vivo) ===\n")
    print("ESPACIO: guardar frame | ESC: salir\n")

    cap = cv2.VideoCapture(camara)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara {camara}.")
        print("Si tienes otra camara, cambia CAMARA en filtros/comun.py")
        return

    titulo = f"{nombre} | ESPACIO: guardar | ESC: salir"
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resultado = filtro(frame)
            cv2.imshow(titulo, resultado)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key == 32:
                sello = datetime.now().strftime("%Y%m%d_%H%M%S")
                ruta = f"{OUTPUT_DIR}/{nombre.replace(' ', '_')}_{sello}.png"
                cv2.imwrite(ruta, resultado)
                print(f"Guardado en: {ruta}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

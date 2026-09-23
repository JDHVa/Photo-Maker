"""Reproduccion de audio sin dependencias, usando MCI de Windows (winmm.dll).

winsound solo toca WAV, y no hay playsound/pygame instalados. MCI (Media Control
Interface) es parte de Windows y reproduce MP3 de forma nativa y asincrona, asi
que el sonido no bloquea el bucle de la camara.

Para el 6-7 hay dos audios en la raiz y se elige uno al azar en cada deteccion.
"""

import os
import random
import threading

RAIZ = os.path.dirname(os.path.abspath(__file__))
AUDIOS_67 = [
    os.path.join(RAIZ, "67-harry.mp3"),
    os.path.join(RAIZ, "67-Meme.mp3"),
]

_ALIAS = "photomake67"
_lock = threading.Lock()

try:
    from ctypes import windll
    _mciw = windll.winmm.mciSendStringW
except Exception:  # no-Windows o sin winmm: el audio queda desactivado
    _mciw = None


def _mci(comando):
    return _mciw(comando, None, 0, None) if _mciw else 1


def reproducir(ruta):
    """Toca un archivo de audio de forma asincrona. Corta el que sonara antes."""
    if _mciw is None or not os.path.exists(ruta):
        return
    with _lock:
        _mci(f"close {_ALIAS}")
        # 'mpegvideo' es el dispositivo que MCI usa para MP3; si fallara, se
        # deja que deduzca el tipo por la extension del archivo.
        if _mci(f'open "{ruta}" type mpegvideo alias {_ALIAS}') != 0:
            if _mci(f'open "{ruta}" alias {_ALIAS}') != 0:
                return
        _mci(f"play {_ALIAS}")


def reproducir_67():
    """Elige uno de los audios del 6-7 al azar y lo reproduce."""
    disponibles = [a for a in AUDIOS_67 if os.path.exists(a)]
    if disponibles:
        reproducir(random.choice(disponibles))

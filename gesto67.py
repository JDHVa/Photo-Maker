MUNECA = 0
PUNTAS = (8, 12, 16, 20)
NUDILLOS = (5, 9, 13, 17)

DEDOS_ABIERTOS = 2      # ya no se exige la mano muy abierta
SEPARACION_MIN = 0.10   # manos separadas en horizontal
UMBRAL = 0.09           # cuanto hay que subir/bajar para contar (mover MUCHO)
CAMBIOS = 2             # cuantos vaivenes hacen falta (arriba-abajo)
VENTANA_MS = 3000       # margen de tiempo para hacerlos
OLVIDO_MS = 1200
ESPERA_MS = 1500
EXIGIR_ABIERTA = False  # True vuelve a pedir las manos abiertas


def _mano_abierta(mano):
    mx, my = mano[MUNECA].x, mano[MUNECA].y
    estirados = 0
    for punta, nudillo in zip(PUNTAS, NUDILLOS):
        d_punta = (mano[punta].x - mx) ** 2 + (mano[punta].y - my) ** 2
        d_nudillo = (mano[nudillo].x - mx) ** 2 + (mano[nudillo].y - my) ** 2
        if d_punta > d_nudillo:
            estirados += 1
    return estirados >= DEDOS_ABIERTOS


def _centro(mano):
    idx = (MUNECA,) + NUDILLOS
    return (
        sum(mano[i].x for i in idx) / len(idx),
        sum(mano[i].y for i in idx) / len(idx),
    )


class Gesto67:
    def __init__(self):
        self.signo = 0
        self.cambios = 0
        self.primer_cambio = 0
        self.ultimo_ok = 0
        self.ultima_deteccion = -ESPERA_MS
        self.progreso = 0.0
        self.diag = "sin manos"  # texto para depurar en pantalla

    def reiniciar(self):
        self.signo = 0
        self.cambios = 0
        self.primer_cambio = 0
        self.progreso = 0.0

    def actualizar(self, manos, ms):
        if not self._postura_valida(manos):
            if len(manos) < 2:
                self.diag = f"veo {len(manos)} mano(s), hacen falta 2"
            else:
                sep = abs(_centro(manos[0])[0] - _centro(manos[1])[0])
                self.diag = f"separa mas las manos ({sep:.2f}<{SEPARACION_MIN})"
            if ms - self.ultimo_ok > OLVIDO_MS:
                self.reiniciar()
            return False

        self.ultimo_ok = ms
        izq, der = sorted((_centro(m) for m in manos[:2]))
        desnivel = izq[1] - der[1]

        if abs(desnivel) < UMBRAL:
            self.diag = f"sube/baja mas ({abs(desnivel):.2f}<{UMBRAL})  cambios {self.cambios}/{CAMBIOS}"
            return False

        self.diag = f"desnivel {desnivel:+.2f}  cambios {self.cambios}/{CAMBIOS}"

        signo = 1 if desnivel > 0 else -1
        if signo == self.signo:
            return False

        if self.signo == 0:
            self.signo = signo
            return False

        self.signo = signo
        if self.cambios == 0 or ms - self.primer_cambio > VENTANA_MS:
            self.cambios = 1
            self.primer_cambio = ms
        else:
            self.cambios += 1
        self.progreso = min(1.0, self.cambios / CAMBIOS)

        if self.cambios >= CAMBIOS and ms - self.ultima_deteccion > ESPERA_MS:
            self.ultima_deteccion = ms
            self.reiniciar()
            return True
        return False

    def _postura_valida(self, manos):
        if len(manos) < 2:
            return False
        a, b = manos[0], manos[1]
        if EXIGIR_ABIERTA and not (_mano_abierta(a) and _mano_abierta(b)):
            return False

        return abs(_centro(a)[0] - _centro(b)[0]) >= SEPARACION_MIN

    def celebrando(self, ms):
        return ms - self.ultima_deteccion < ESPERA_MS

"""Registro unico de filtros, compartido por menu.py y main.py."""

import importlib.util
import os

FILTROS_DIR = os.path.dirname(os.path.abspath(__file__))

# (tecla, nombre, archivo, funcion, descripcion)
FILTROS = [
    ("1", "Suave", "filtro-suave.py", "filtro_spiderverse",
     "colores planos saturados con bordes de tinta"),
    ("2", "Comic", "filtro-comic.py", "filtro_comic",
     "pagina impresa: puntos Ben-Day, papel, tinta roja y marco"),
    ("3", "Anime", "filtro-anime.py", "filtro_anime",
     "piel lisa, paleta pastel, lineas finas y bloom"),
    ("4", "Manga", "filtro-manga.py", "filtro_manga",
     "blanco y negro puro con screentones y lineas de velocidad"),
    ("5", "Pixel Art", "filtro-pixel.py", "filtro_pixel",
     "16 colores retro con dithering ordenado"),
    ("6", "Sin City", "filtro-sincity.py", "filtro_sincity",
     "blanco y negro extremo, solo los rojos sobreviven"),
]


def cargar(archivo, funcion):
    """Los archivos tienen guion en el nombre, asi que no se pueden importar
    con `import`; hay que cargarlos por ruta."""
    ruta = os.path.join(FILTROS_DIR, archivo)
    spec = importlib.util.spec_from_file_location(archivo[:-3], ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return getattr(modulo, funcion)


def cargar_todos():
    """Carga los 6 filtros de una vez. Se hace al arrancar, nunca por frame."""
    return {tecla: (nombre, cargar(arch, fn))
            for tecla, nombre, arch, fn, _ in FILTROS}


def buscar(tecla):
    return next((f for f in FILTROS if f[0] == tecla), None)

"""PhotoMake - menu de filtros a pantalla completa.

Para el modo de ventana con las manos, usa main.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "filtros"))

import comun
import registro


def main():
    print("\n=== PhotoMake ===\n")
    for tecla, nombre, _, _, desc in registro.FILTROS:
        print(f"  [{tecla}] {nombre:<10} {desc}")

    elegido = registro.buscar(input("\nElige un filtro: ").strip())
    if elegido is None:
        print("Opcion no valida.")
        return

    _, nombre, archivo, funcion, _ = elegido
    comun.correr(nombre, registro.cargar(archivo, funcion))


if __name__ == "__main__":
    main()

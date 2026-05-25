"""
main.py
-------
Punto de entrada principal del proyecto.

Uso:
    python main.py --mode [demo | benchmark | all]

Opciones:
    demo      → genera GIFs de patrones clásicos + aleatorio
    benchmark → corre el benchmark y guarda las gráficas
    all       → ejecuta ambos (por defecto)
"""

import argparse
import os

from visualize  import save_all_pattern_gifs
from benchmark  import main as run_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Juego de la Vida de Conway — Tarea 1"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "benchmark", "all"],
        default="all",
        help="Qué ejecutar: demo (GIFs), benchmark (rendimiento), all (ambos)"
    )
    parser.add_argument(
        "--grid", type=int, default=48,
        help="Tamaño de grilla para los GIFs de patrones (default: 48)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs("outputs/gifs",  exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    if args.mode in ("demo", "all"):
        print("\n── Generando visualizaciones ──")
        save_all_pattern_gifs(
            output_dir="outputs/gifs",
            grid_size=args.grid,
            n_frames=60
        )

    if args.mode in ("benchmark", "all"):
        print("\n── Ejecutando benchmark ──")
        run_benchmark(output_dir="outputs/plots")

    print("\n¡Listo! Revisa la carpeta outputs/")


if __name__ == "__main__":
    main()

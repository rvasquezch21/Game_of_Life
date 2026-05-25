"""
benchmark.py
------------
Medición empírica del rendimiento del Juego de la Vida.

Qué hace:
  1. Ejecuta la simulación para grillas de distintos tamaños.
  2. Mide el tiempo promedio por iteración.
  3. Grafica tiempo vs número de celdas (escala lineal y log-log).
  4. Superpone curvas teóricas de complejidad para comparar.
  5. Guarda las gráficas en outputs/plots/.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import time

from game_of_life import GameOfLife, benchmark_step


# ── Configuración del experimento ─────────────────────────────────────────────

# Tamaños de grilla a evaluar (lado N → N² celdas)
GRID_SIZES = [32, 64, 128, 256, 512, 1024]

# Pasos por medición (más pasos → promedio más estable)
N_STEPS = 30

# Repeticiones para estimar varianza
N_REPS = 3


# ── Ejecución del benchmark ───────────────────────────────────────────────────

def run_benchmark(sizes=GRID_SIZES, n_steps=N_STEPS, n_reps=N_REPS):
    """
    Mide el tiempo medio por paso para cada tamaño de grilla.

    Returns
    -------
    cell_counts : np.ndarray — número de celdas (N²) para cada tamaño.
    mean_times  : np.ndarray — tiempo promedio en segundos por paso.
    std_times   : np.ndarray — desviación estándar entre repeticiones.
    """
    cell_counts = np.array([s * s for s in sizes], dtype=np.float64)
    mean_times  = np.zeros(len(sizes))
    std_times   = np.zeros(len(sizes))

    print(f"{'Grilla':>10}  {'Celdas':>10}  {'T_prom (ms)':>12}  {'σ (ms)':>8}")
    print("-" * 48)

    for i, size in enumerate(sizes):
        reps = []
        for _ in range(n_reps):
            t = benchmark_step(size, n_steps)
            reps.append(t)
        mean_times[i] = np.mean(reps)
        std_times[i]  = np.std(reps)
        print(
            f"{size:>5}x{size:<5}  "
            f"{int(cell_counts[i]):>10,}  "
            f"{mean_times[i]*1000:>12.3f}  "
            f"{std_times[i]*1000:>8.3f}"
        )

    return cell_counts, mean_times, std_times


# ── Curvas teóricas de referencia ─────────────────────────────────────────────

def _fit_curves(cell_counts, mean_times):
    """
    Genera curvas O(n), O(n log n) y O(n²) escaladas al primer punto medido.
    Esto permite comparar visualmente la pendiente sin conocer la constante.
    """
    n = cell_counts
    # Factor de escala: igualamos cada curva al primer punto medido
    c_linear  = mean_times[0] / n[0]
    c_nlogn   = mean_times[0] / (n[0] * np.log(n[0]))
    c_n2      = mean_times[0] / (n[0] ** 2)

    return {
        "O(n)"       : c_linear * n,
        "O(n log n)" : c_nlogn  * n * np.log(n),
        "O(n²)"      : c_n2     * n ** 2,
    }


# ── Graficación ───────────────────────────────────────────────────────────────

STYLE = {
    "measured" : dict(color="#00d4ff", marker="o", linewidth=2,
                      markersize=7, label="Medido", zorder=5),
    "O(n)"     : dict(color="#90ee90", linestyle="--", linewidth=1.4,
                      alpha=0.85, label="O(n)"),
    "O(n log n)":dict(color="#ffd700", linestyle="-.", linewidth=1.4,
                      alpha=0.85, label="O(n log n)"),
    "O(n²)"    : dict(color="#ff6b6b", linestyle=":",  linewidth=1.4,
                      alpha=0.85, label="O(n²)"),
}

DARK_BG = "#1a1a2e"
GRID_COLOR = "#2e2e4e"


def _apply_dark_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(DARK_BG)
    ax.figure.patch.set_facecolor(DARK_BG)
    ax.set_title(title,  color="white", fontsize=13, pad=10)
    ax.set_xlabel(xlabel, color="#aaa", fontsize=11)
    ax.set_ylabel(ylabel, color="#aaa", fontsize=11)
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7)
    leg = ax.legend(facecolor="#0f0f1f", edgecolor=GRID_COLOR,
                    labelcolor="white", fontsize=10)


def plot_linear(cell_counts, mean_times, std_times, curves, save_path=None):
    """Gráfica en escala lineal con barras de error y curvas teóricas."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Curvas teóricas
    for name, y in curves.items():
        ax.plot(cell_counts, y, **STYLE[name])

    # Datos medidos con barras de error (±1σ)
    ax.errorbar(cell_counts, mean_times,
                yerr=std_times,
                **STYLE["measured"])

    _apply_dark_style(
        ax,
        title="Tiempo por iteración vs Número de celdas (escala lineal)",
        xlabel="Número de celdas (N²)",
        ylabel="Tiempo promedio por paso (s)"
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=DARK_BG)
        print(f"  Gráfica lineal guardada: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_loglog(cell_counts, mean_times, std_times, curves, save_path=None):
    """
    Gráfica log-log.

    En escala log-log una función O(nᵏ) aparece como una línea recta con
    pendiente k, lo que hace muy fácil identificar el orden de complejidad.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, y in curves.items():
        ax.loglog(cell_counts, y, **STYLE[name])

    ax.errorbar(cell_counts, mean_times,
                yerr=std_times,
                **STYLE["measured"])

    _apply_dark_style(
        ax,
        title="Tiempo por iteración vs Número de celdas (escala log-log)",
        xlabel="Número de celdas (N²)  [log]",
        ylabel="Tiempo promedio por paso (s)  [log]"
    )

    # Anotar pendiente empírica
    if len(cell_counts) >= 2:
        log_x = np.log10(cell_counts)
        log_y = np.log10(mean_times)
        slope = np.polyfit(log_x, log_y, 1)[0]
        ax.annotate(
            f"Pendiente empírica ≈ {slope:.2f}",
            xy=(cell_counts[-2], mean_times[-2]),
            xytext=(cell_counts[-3], mean_times[-1] * 2),
            color="white", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="white", lw=1.2)
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=DARK_BG)
        print(f"  Gráfica log-log guardada:  {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_alive_cells(sizes=None, save_path=None):
    """
    Gráfica auxiliar: evolución del número de celdas vivas a lo largo
    de 200 generaciones para una grilla 64×64 aleatoria.
    Muestra cómo el sistema llega a un estado de equilibrio.
    """
    if sizes is None:
        sizes = [64]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor(DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    colors = ["#00d4ff", "#ff6b6b", "#90ee90", "#ffd700"]
    for idx, s in enumerate(sizes):
        game = GameOfLife(s, s)
        counts = [game.alive_count]
        for _ in range(200):
            game.step()
            counts.append(game.alive_count)
        ax.plot(counts, color=colors[idx % len(colors)],
                linewidth=1.5, label=f"{s}×{s}")

    _apply_dark_style(
        ax,
        title="Evolución de celdas vivas (primeras 200 generaciones)",
        xlabel="Generación",
        ylabel="Celdas vivas"
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=DARK_BG)
        print(f"  Gráfica de celdas vivas:   {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ── Función principal ─────────────────────────────────────────────────────────

def main(output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 48)
    print("  Benchmark — Juego de la Vida de Conway")
    print("=" * 48)

    # 1. Correr el benchmark
    cell_counts, mean_times, std_times = run_benchmark()

    # 2. Calcular curvas teóricas
    curves = _fit_curves(cell_counts, mean_times)

    # 3. Graficar
    print("\nGenerando gráficas ...")
    plot_linear(cell_counts, mean_times, std_times, curves,
                save_path=os.path.join(output_dir, "benchmark_linear.png"))
    plot_loglog(cell_counts, mean_times, std_times, curves,
                save_path=os.path.join(output_dir, "benchmark_loglog.png"))
    plot_alive_cells([32, 64, 128],
                     save_path=os.path.join(output_dir, "alive_cells_evolution.png"))

    # 4. Resumen de análisis
    log_x  = np.log10(cell_counts)
    log_y  = np.log10(mean_times)
    slope  = np.polyfit(log_x, log_y, 1)[0]
    print(f"\n── Análisis de complejidad empírica ──")
    print(f"  Pendiente log-log ≈ {slope:.3f}")
    if slope < 1.1:
        label = "O(n) — lineal ✓"
    elif slope < 1.6:
        label = "O(n log n) — casi lineal"
    else:
        label = "O(n²) — cuadrática"
    print(f"  Clase de complejidad estimada: {label}")
    print(f"\n✓ Experimento completo. Resultados en '{output_dir}/'")


if __name__ == "__main__":
    main()

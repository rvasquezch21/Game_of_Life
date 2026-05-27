import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from game_of_life import GameOfLife


# Paleta de colores
CMAP = "binary"  


def _build_figure(title: str, grid_size: tuple):
    """Crea figura y ejes con estilo consistente."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def animate_pattern(pattern_name: str,
                    grid_size: int = 32,
                    n_frames: int = 60,
                    interval_ms: int = 120,
                    save_path: str = None):
    game = GameOfLife(grid_size, grid_size)
    offset = grid_size // 4     
    game.set_pattern(pattern_name, offset_row=offset, offset_col=offset)

    fig, ax = _build_figure(
        f"Conway's Game of Life — {pattern_name.capitalize()}",
        (grid_size, grid_size)
    )

    img = ax.imshow(game.get_state(), cmap=CMAP, vmin=0, vmax=1,
                    interpolation="nearest")

    # Etiqueta de generación
    gen_text = ax.text(0.02, 0.97, "Gen: 0", transform=ax.transAxes,
                       color="lime", fontsize=10, va="top")

    def update(frame):
        """Función llamada por FuncAnimation en cada fotograma."""
        game.step()
        img.set_data(game.get_state())
        gen_text.set_text(f"Gen: {game.generation}  |  Vivas: {game.alive_count}")
        return [img, gen_text]

    anim = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval_ms,
        blit=True
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval_ms)
        print(f"  Animación guardada en: {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return anim   


def animate_random(rows: int = 64,
                   cols: int = 64,
                   n_frames: int = 100,
                   interval_ms: int = 80,
                   save_path: str = None):

    game = GameOfLife(rows, cols) 

    fig, ax = _build_figure(
        f"Conway's Game of Life — Random {rows}×{cols}",
        (rows, cols)
    )

    img = ax.imshow(game.get_state(), cmap=CMAP, vmin=0, vmax=1,
                    interpolation="nearest")
    gen_text = ax.text(0.02, 0.97, "Gen: 0", transform=ax.transAxes,
                       color="lime", fontsize=9, va="top")

    def update(frame):
        game.step()
        img.set_data(game.get_state())
        gen_text.set_text(f"Gen: {game.generation}  |  Vivas: {game.alive_count}")
        return [img, gen_text]

    anim = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval_ms,
        blit=True
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval_ms)
        print(f"  Animación guardada en: {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return anim


def save_all_pattern_gifs(output_dir: str = "outputs/gifs",
                          grid_size: int = 48,
                          n_frames: int = 60):
    os.makedirs(output_dir, exist_ok=True)
    patterns = ["glider", "blinker", "toad", "block", "beacon", "pulsar"]
    for name in patterns:
        path = os.path.join(output_dir, f"{name}.gif")
        print(f"Generando {name}.gif ...")
        animate_pattern(name,
                        grid_size=grid_size,
                        n_frames=n_frames,
                        interval_ms=120,
                        save_path=path)
    animate_random(rows=64, cols=64,
                   n_frames=80,
                   save_path=os.path.join(output_dir, "random_64x64.gif"))
    print("✓ Todos los GIFs generados.")


# Ejecución 
if __name__ == "__main__":
    save_all_pattern_gifs()

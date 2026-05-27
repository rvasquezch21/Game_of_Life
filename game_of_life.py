import numpy as np
from scipy.signal import convolve2d
import time


# Kernel de convolución: cuenta los 8 vecinos de cada celda
# (el centro es 0 para no contar la celda misma)
NEIGHBOR_KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=np.uint8)


# Patrones clásicos 

PATTERNS = {
    "glider": [
        (0, 1),
        (1, 2),
        (2, 0), (2, 1), (2, 2),
    ],
    "blinker": [
        (1, 0), (1, 1), (1, 2),
    ],
    "toad": [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2),
    ],
    "block": [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
    ],
    "beacon": [
        (0, 0), (0, 1),
        (1, 0),
        (2, 3),
        (3, 2), (3, 3),
    ],
    "pulsar": [
        (0,2),(0,3),(0,4),(0,8),(0,9),(0,10),
        (2,0),(2,5),(2,7),(2,12),
        (3,0),(3,5),(3,7),(3,12),
        (4,0),(4,5),(4,7),(4,12),
        (5,2),(5,3),(5,4),(5,8),(5,9),(5,10),
        (7,2),(7,3),(7,4),(7,8),(7,9),(7,10),
        (8,0),(8,5),(8,7),(8,12),
        (9,0),(9,5),(9,7),(9,12),
        (10,0),(10,5),(10,7),(10,12),
        (12,2),(12,3),(12,4),(12,8),(12,9),(12,10),
    ],
}


class GameOfLife:

    def __init__(self, rows: int, cols: int, initial_state=None):
        self.rows = rows
        self.cols = cols
        self.generation = 0       

        if initial_state is not None:
            self.grid = np.array(initial_state, dtype=np.uint8)
            assert self.grid.shape == (rows, cols), (
                f"initial_state debe tener forma ({rows}, {cols}), "
                f"pero tiene {self.grid.shape}"
            )
        else:
            rng = np.random.default_rng()
            self.grid = rng.choice(
                [0, 1], size=(rows, cols), p=[0.7, 0.3]
            ).astype(np.uint8)

    # Métodos públicos requeridos

    def step(self) -> None:

        neighbors = convolve2d(self.grid, NEIGHBOR_KERNEL,
                               mode='same', boundary='wrap')

        survive = (self.grid == 1) & ((neighbors == 2) | (neighbors == 3))
        born    = (self.grid == 0) & (neighbors == 3)

        self.grid = (survive | born).astype(np.uint8)
        self.generation += 1

    def run(self, steps: int) -> None:
 
        for _ in range(steps):
            self.step()

    def get_state(self) -> np.ndarray:
        return self.grid.copy()

    def set_pattern(self, pattern_name: str,
                    offset_row: int = 0, offset_col: int = 0) -> None:

        if pattern_name not in PATTERNS:
            raise ValueError(
                f"Patrón '{pattern_name}' no reconocido. "
                f"Disponibles: {list(PATTERNS.keys())}"
            )
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for r, c in PATTERNS[pattern_name]:
            row = (r + offset_row) % self.rows
            col = (c + offset_col) % self.cols
            self.grid[row, col] = 1
        self.generation = 0

    @property
    def alive_count(self) -> int:
        """Número de celdas vivas en la generación actual."""
        return int(self.grid.sum())


# Utilidad de benchmark 

def benchmark_step(size: int, n_steps: int = 20) -> float:

    game = GameOfLife(size, size)      
    game.step()
    start = time.perf_counter()
    game.run(n_steps)
    elapsed = time.perf_counter() - start
    return elapsed / n_steps

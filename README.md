# Tarea 1 — Juego de la Vida de Conway

**Curso:** Computación Paralela y Distribuida  
**Universidad:** LEAD University  
**Profesor:** Johansell Villalobos Cubillo  

---

## Descripción

Implementación orientada a objetos del Juego de la Vida de Conway en Python.  
El proyecto incluye:

- Clase `GameOfLife` con lógica vectorizada (NumPy + SciPy).
- Visualizaciones animadas (GIF) de patrones clásicos.
- Benchmark empírico de rendimiento con gráficas de complejidad.

---

## Estructura del proyecto

```
game_of_life/
├── game_of_life.py   # Clase principal + función de benchmark
├── visualize.py      # Animaciones con matplotlib
├── benchmark.py      # Medición de rendimiento y gráficas
├── main.py           # Punto de entrada (CLI)
├── requirements.txt  # Dependencias
├── README.md
└── outputs/
    ├── gifs/         # Animaciones generadas
    └── plots/        # Gráficas de rendimiento
```

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/<tu-usuario>/game_of_life.git
cd game_of_life

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución

### Todo (visualizaciones + benchmark)
```bash
python main.py --mode all
```

### Solo las animaciones GIF
```bash
python main.py --mode demo --grid 48
```

### Solo el benchmark de rendimiento
```bash
python main.py --mode benchmark
```

### Uso directo de los módulos

```python
from game_of_life import GameOfLife

# Estado aleatorio
game = GameOfLife(64, 64)
game.run(100)
state = game.get_state()   # numpy array 64×64

# Patrón específico
game2 = GameOfLife(32, 32)
game2.set_pattern("glider", offset_row=5, offset_col=5)
for _ in range(50):
    game2.step()
    print(game2.alive_count)
```

---

## Reglas de Conway implementadas

| Condición | Resultado |
|-----------|-----------|
| Celda viva con < 2 vecinos vivos | Muere (soledad) |
| Celda viva con 2 ó 3 vecinos vivos | Sobrevive |
| Celda viva con > 3 vecinos vivos | Muere (superpoblación) |
| Celda muerta con exactamente 3 vecinos vivos | Nace |

---

## Patrones disponibles

| Patrón | Tipo | Descripción |
|--------|------|-------------|
| `glider` | Nave espacial | Se desplaza diagonalmente |
| `blinker` | Oscilador (período 2) | Alterna entre horizontal y vertical |
| `toad` | Oscilador (período 2) | Patrón de 6 celdas |
| `block` | Estático | 2×2, no cambia |
| `beacon` | Oscilador (período 2) | Dos bloques que parpadean |
| `pulsar` | Oscilador (período 3) | Gran patrón simétrico |

---

## Resultados de rendimiento

Los experimentos muestran que la implementación escala aproximadamente como **O(n)**,
donde n es el número de celdas. Esto se debe al uso de `scipy.signal.convolve2d` que
aprovecha FFT internamente para grillas grandes.

Ver gráficas en `outputs/plots/`.

---

## Decisiones de diseño

- **NumPy vectorizado:** En lugar de iterar celda por celda (O(n) con constante alta),
  se aplica una convolución 2D sobre toda la grilla simultáneamente.
- **`scipy.signal.convolve2d`:** Cuenta los 8 vecinos de cada celda en una sola llamada.
  Con `boundary='wrap'` el tablero es toroidal (los bordes se conectan).
- **`dtype=uint8`:** Ocupa 1 byte por celda (vs 8 bytes con float64), reduciendo el uso
  de memoria en 8×.

---

## Dependencias

```
numpy
scipy
matplotlib
pillow
```

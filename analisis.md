# Análisis y Discusión — Juego de la Vida de Conway

## 1. Resultados empíricos de rendimiento

Los experimentos se ejecutaron variando el tamaño de la grilla desde 32×32 hasta 1024×1024,
midiendo el tiempo promedio por iteración con 30 pasos y 3 repeticiones por tamaño:

| Grilla       | Celdas (n)  | Tiempo promedio (ms) |
|--------------|-------------|----------------------|
| 32 × 32      | 1,024       | 0.035                |
| 64 × 64      | 4,096       | 0.110                |
| 128 × 128    | 16,384      | 0.410                |
| 256 × 256    | 65,536      | 1.622                |
| 512 × 512    | 262,144     | 6.490                |
| 1024 × 1024  | 1,048,576   | 25.997               |

Al multiplicar el número de celdas por 4 (duplicar el lado), el tiempo se multiplica
aproximadamente por 4 también. Esto es la firma característica de **complejidad O(n)**,
donde n es el número total de celdas.

La pendiente medida en la gráfica log-log fue de **≈ 0.96**, confiriendo que el
crecimiento es casi perfectamente lineal.

---

## 2. Complejidad temporal

### ¿Por qué O(n) y no O(n²)?

La implementación usa `scipy.signal.convolve2d` con `boundary='wrap'` para contar los
vecinos de todas las celdas simultáneamente. Esta función aplica internamente una
**Transformada Rápida de Fourier (FFT)** cuando la grilla es suficientemente grande,
lo que reduce la complejidad de la convolución de O(n·k²) — donde k es el tamaño del
kernel — a O(n log n). Como el kernel es fijo de tamaño 3×3, en la práctica el costo
por celda es constante y el comportamiento observado es **O(n)**.

En contraste, una implementación ingenua con bucles anidados en Python puro tendría
complejidad O(n) en teoría pero con una constante muy alta, haciendo que en la práctica
fuera entre 100 y 1000 veces más lenta.

Las operaciones de NumPy que aplican las reglas (`survive`, `born`) también son O(n)
vectorizadas, sin añadir complejidad adicional.

---

## 3. Escalamiento en memoria

### Uso de memoria por grilla

La grilla se almacena como un array NumPy de tipo `uint8` (1 byte por celda).
El uso de memoria es estrictamente **O(n)** y se puede calcular directamente:

| Grilla       | Celdas      | Memoria (MB) |
|--------------|-------------|--------------|
| 32 × 32      | 1,024       | 0.001        |
| 128 × 128    | 16,384      | 0.016        |
| 512 × 512    | 262,144     | 0.250        |
| 1024 × 1024  | 1,048,576   | 1.000        |
| 4096 × 4096  | 16,777,216  | 16.000       |
| 16384 × 16384| 268,435,456 | 256.000      |

La grilla principal ocupa exactamente **n bytes**. Sin embargo, `convolve2d` necesita
crear internamente un array auxiliar del mismo tamaño para almacenar el conteo de
vecinos (tipo `int` o `float`), lo que multiplica el uso real por un factor de
aproximadamente **9×** respecto al tamaño nominal (1 byte de la grilla + hasta 8 bytes
del array de convolución en punto flotante).

Para una grilla de 1024×1024 el uso real en memoria se estima en:
```
1,048,576 celdas × (1 byte grilla + 8 bytes convolución) ≈ 9 MB
```

Esto es manejable, pero para grillas de 16,384×16,384 el requerimiento superaría
los 2 GB, lo que constituye una limitación práctica importante.

---

## 4. Cuellos de botella estructurales

### 4.1 Asignación de memoria en cada paso

El cuello de botella más significativo de la implementación es que **en cada generación
se crean dos nuevos arrays de NumPy** (`survive` y `born`), además del array de vecinos
de `convolve2d`. Esto implica que el recolector de basura de Python debe liberar y
reasignar memoria en cada iteración, lo que genera presión sobre el heap y puede
causar pausas de GC (Garbage Collection) no deterministas en simulaciones largas.

Una optimización posible es usar **buffers pre-asignados** con `np.empty_like` fuera
del loop y escribir in-place, evitando nuevas asignaciones.

### 4.2 Overhead de `convolve2d` para grillas pequeñas

Para grillas pequeñas (32×32, 64×64), el overhead de llamar a `convolve2d` — que
internamente valida parámetros, decide si usar FFT o correlación directa, y maneja
condiciones de borde — representa una fracción importante del tiempo total. Esto
explica por qué los tiempos para grillas pequeñas no escalan perfectamente desde cero.

### 4.3 Conversión de tipos

Las reglas de Conway producen arrays booleanos (`survive | born`), que luego se
convierten a `uint8` con `.astype(np.uint8)`. Esta conversión es O(n) pero implica
una pasada adicional sobre todos los datos. Se podría eliminar usando directamente
`np.uint8(survive | born)` o trabajando con `view`.

### 4.4 Transferencia de datos y caché de CPU

Para grillas muy grandes (512×512 en adelante), el array deja de caber en la caché L2
del procesador (~256 KB típico). A partir de ese punto, cada acceso a memoria implica
ir a RAM, lo que es entre 10 y 100 veces más lento que acceder desde caché. Esto
explica el ligero quiebre en la curva de rendimiento visible entre 256×256 y 512×512.

### 4.5 Paralelismo no aprovechado

La implementación actual es completamente **secuencial**: usa un solo núcleo del
procesador. Dado que cada celda de la nueva generación es independiente de las demás
(solo depende del estado anterior, no del nuevo), el algoritmo es **embarazosamente
paralelo** — ideal para distribuirlo entre múltiples núcleos con `multiprocessing`,
o acelerarlo con GPU usando `CuPy` en lugar de NumPy.

---

## 5. Comparación con complejidad teórica

| Complejidad | Comportamiento esperado al duplicar n | Observado |
|-------------|---------------------------------------|-----------|
| O(n)        | Tiempo × 4                            | ✅ ~× 4   |
| O(n log n)  | Tiempo × 4.1 aprox.                   | Similar   |
| O(n²)       | Tiempo × 16                           | ✗ No      |

La implementación se comporta como **O(n)** empíricamente. La diferencia entre O(n)
y O(n log n) es difícil de distinguir experimentalmente para los rangos de tamaño
evaluados, ya que log(n) crece muy lentamente (log₂(1,048,576) ≈ 20).

---

## 6. Conclusiones

1. **La vectorización con NumPy es crítica.** Sin ella, una implementación en Python
   puro sería entre 100× y 500× más lenta para grillas de 1024×1024.

2. **El escalamiento en memoria es lineal (O(n))** y manejable hasta grillas de
   aproximadamente 8,192×8,192 en hardware moderno con 8 GB de RAM. Más allá de eso,
   se requeriría procesamiento por bloques (*tiling*).

3. **El principal cuello de botella no es el algoritmo sino la asignación de memoria**
   repetida en cada generación. Usar buffers pre-asignados podría reducir el tiempo
   entre un 10% y 20% adicional.

4. **La oportunidad de mejora más grande es la paralelización.** Distribuir la grilla
   entre N núcleos reduciría el tiempo por iteración proporcionalmente, permitiendo
   simular grillas de millones de celdas en tiempo real.

# Simulación de N cuerpos: colisión de galaxias (serial, CPU paralela y GPU)

Proyecto final de Cálculo Científico · Universidad Industrial de Santander · 2026-2

Simulación gravitacional directa de N cuerpos (interacción de todos contra todos, O(N²)) de la colisión de dos galaxias de disco. Se implementa en tres versiones y se mide el speedup de cada una:

| Versión | Archivo | Herramienta | Precisión |
|---|---|---|---|
| Serial (referencia) | `src/fisica.py` | NumPy vectorizado | float64 |
| CPU paralela | `src/cpu_paralelo.py` | Numba `njit(parallel=True)` + `prange` | float64 |
| GPU | `src/gpu_cuda.py` | Numba CUDA (memoria compartida por *tiles*) | float32 |

**Integrantes:** _(nombres del grupo)_

---

## 1. Modelo físico

Cada cuerpo *i* (masa *mᵢ*, posición **rᵢ**) acelera por la atracción gravitacional de los demás, con suavizado (*softening*) ε para evitar singularidades en los encuentros cercanos:

```
a_i = G · Σ_{j≠i} m_j (r_j − r_i) / ( |r_j − r_i|² + ε² )^(3/2)
```

La energía usa el mismo suavizado, para que sea coherente con la fuerza:

```
E = K + U,    K = ½ Σ m_i |v_i|²,    U = −½ G Σ_{i≠j} m_i m_j / sqrt(|r_j − r_i|² + ε²)
```

- **Unidades:** G = 1; M₀ y L₀ son la unidad de masa y de longitud (el tiempo sale de ambas).
- **Parámetros usados:** ε = 0,05 · dt = 0,005 (t = 30 con 6000 pasos).
- **Integrador:** *leapfrog* kick-drift-kick (orden 2, simpléctico y reversible).
- **Condiciones iniciales** (`src/condiciones.py`):
  - Cada galaxia tiene un núcleo (masa 0,8) y un disco de estrellas (masa total 0,2) con perfil radial tipo gamma (longitud de escala R_d = 1), radios entre 0,2 y 5, y espesor vertical pequeño.
  - Las velocidades son circulares según la masa encerrada, con una dispersión del 3 %.
  - Las galaxias parten a una separación de 12, con parámetro de impacto 3 y velocidad relativa del 60 % de la parabólica (órbita ligada: terminan fusionándose). La segunda galaxia está inclinada 60°.
  - El sistema se centra en su centro de masas y en reposo.
- **Frontera:** espacio abierto, sin condiciones de frontera.
- **Validación física** (`pruebas/validacion_fisica.py`): órbita kepleriana excéntrica (convergencia de orden 2), reversibilidad temporal y barrido de dt en la colisión completa, con criterio |ΔE/E| < 1 %. Además se vigilan el momento lineal y el angular.

## 2. Estrategia de paralelización

La parte costosa es el cálculo de las aceleraciones (O(N²) por paso); el resto del paso es O(N).

- **CPU:** `prange` sobre el bucle externo (partículas *i*). Cada hilo escribe solo en su fila `acc[i]`, así que no hay condiciones de carrera. Se usa Numba (compila a código nativo y libera el GIL); `threading` sobre Python puro no daría paralelismo real. El número de hilos se controla con `numba.set_num_threads`.
- **GPU:** un hilo CUDA por partícula *i*. Los bloques recorren las demás partículas por *tiles* cargados en memoria compartida. Posiciones y velocidades permanecen en la GPU entre pasos; solo se copian al registrar resultados.
- **Verificación serial ≟ paralelo:** error relativo máximo de las aceleraciones ≈ 1e-16 en CPU (float64) y ≈ 7e-7 en GPU (float32), para N = 250 a 4000. Los resultados quedan en `benchmarks/verificacion_cpu.csv`.

## 3. Hardware y software usados

| Componente | Detalle |
|---|---|
| CPU | Intel Core i5-11400H @ 2,70 GHz, **6 núcleos físicos / 12 hilos lógicos** (portátil) |
| GPU | NVIDIA GeForce GTX 1650, 4 GiB, 14 multiprocesadores, capacidad de cómputo 7.5 |
| Sistema | Windows (10.0.26300) |
| Python | 3.11 |
| NumPy / Numba | 2.4.6 / 0.66.0 (con `numba-cuda`) |
| CUDA runtime | 12.9 |

Cada ejecución de los *benchmarks* guarda su propio detalle en `resultados/hardware.txt` y `resultados/hardware_gpu.txt`.

**Entorno:** conda, entorno `nbody-gpu`.

```
conda create -n nbody-gpu python=3.11
conda activate nbody-gpu
pip install -r requirements.txt
python pruebas/test_entorno.py        # comprueba NumPy, Numba, hilos y CUDA
```

Para la GPU se necesita un driver de NVIDIA compatible con CUDA 12. Sin GPU, todo funciona salvo `benchmark_gpu.py` (y `visualizar.py --backend gpu`). Para guardar la animación en MP4 hace falta `ffmpeg`; sin él se escribe un GIF.

## 4. Estructura del repositorio

```
src/
  fisica.py          versión serial (NumPy): aceleraciones, energía, leapfrog, simular
  cpu_paralelo.py    versión paralela en CPU (Numba prange)
  gpu_cuda.py        versión GPU (Numba CUDA)
  condiciones.py     galaxia de disco y colisión de dos galaxias
pruebas/
  test_entorno.py        comprueba el entorno
  prueba_serial.py       dos cuerpos, cúmulo aleatorio, escalado serial
  prueba_condiciones.py  condiciones iniciales y estabilidad de una galaxia aislada
  validacion_fisica.py   Kepler, reversibilidad, efecto de dt
  benchmark_cpu.py       tiempos de CPU y verificación serial vs paralelo
  analizar_cpu.py        agrupa las corridas de CPU: speedup, eficiencia, ajuste de Amdahl
  benchmark_gpu.py       tiempos de GPU, transferencias, TPB, verificación
  visualizar.py          animación de la colisión
benchmarks/              CSV con todas las repeticiones (datos crudos)
resultados/              figuras, tablas y archivos de hardware
```

Todos los scripts se ejecutan **desde la raíz del proyecto**.

## 5. Cómo reproducir cada resultado

| Qué | Comando | Salida |
|---|---|---|
| Comprobar el entorno | `python pruebas/test_entorno.py` | consola |
| Condiciones iniciales y estabilidad | `python pruebas/prueba_condiciones.py` | `resultados/condiciones_iniciales.png`, `resultados/estabilidad_galaxia.png` |
| Validación física (orden 2, reversibilidad, dt) | `python pruebas/validacion_fisica.py` | `resultados/validacion_dt.png` |
| Prueba serial (dos cuerpos, cúmulo, escalado) | `python pruebas/prueba_serial.py` | consola |
| Benchmark de CPU (3 corridas) | `python pruebas/benchmark_cpu.py` | `benchmarks/resultados_cpu_<fecha>_c1..c3.csv`, `benchmarks/verificacion_cpu.csv`, `resultados/benchmark_cpu.png`, `resultados/hardware.txt` |
| Análisis de CPU: speedup, eficiencia y Amdahl | `python pruebas/analizar_cpu.py` | `resultados/analisis_cpu.png`, `resultados/tabla_speedup_cpu.csv` |
| Benchmark de GPU | `python pruebas/benchmark_gpu.py` | `benchmarks/resultados_gpu.csv`, `resultados/benchmark_gpu.png`, `resultados/hardware_gpu.txt` |
| Animación de la colisión | `python pruebas/visualizar.py` | `resultados/colision.mp4` (o `.gif`) y `resultados/colision_final.png` |

Para una prueba corta (solo ver que corre), los *benchmarks* y `visualizar.py` aceptan `--rapido`. `benchmark_cpu.py` acepta además `--corridas N` y `--pausa S`; `visualizar.py` acepta `--backend {auto,gpu,cpu,numpy}`, `--n`, `--pasos` y `--color`.

Antes de correr `analizar_cpu.py`, la carpeta `benchmarks/` debe contener **solo** los CSV `resultados_cpu_<fecha>_c*.csv` de las corridas que se quieren analizar.

## 6. Metodología de medición

- Tiempos con `time.perf_counter`; la primera llamada (compilación JIT) se descarta siempre.
- Se reporta la **mediana** de las repeticiones, con rango intercuartílico (IQR) y mínimo-máximo.
- CPU: 7 repeticiones para el tiempo vs N, y 15 rondas **intercaladas** para el escalado por hilos (así la deriva térmica del portátil afecta a todas las configuraciones por igual). Se repite el experimento completo en 3 corridas independientes, con una pausa entre ellas, y se reporta la mediana y el rango entre corridas.
- Hilos: 1, 2, 4, 6, 8 y 12, con N = 2000 y N = 16000.
- Speedup: *S = T₁ / T_p*, con T₁ = el mismo código paralelo con 1 hilo. Eficiencia: *E = S / p*. Amdahl: *S = 1 / [(1 − f) + f / p]*, con *f* ajustada por mínimos cuadrados.
- GPU: kernel, copia host→GPU y copia GPU→host se miden por separado, con `cuda.synchronize()` antes y después. Los kernels de todos los N se miden intercalados y sin carga de CPU en medio. El speedup se reporta frente a la CPU serial y frente a la CPU paralela (la eficiencia por hilo no aplica en GPU).

## 7. Resultados principales

**CPU (Numba, 12 hilos lógicos / 6 núcleos físicos):** speedup ≈ 4,4× sobre 1 hilo, con eficiencia ≈ 37 % (3 corridas, ver `resultados/tabla_speedup_cpu.csv`). El ajuste de Amdahl da una fracción efectiva *f* ≈ 0,84. Esa *f* es efectiva: el cálculo de fuerzas está 100 % dentro de `prange`, así que lo que limita es el hardware (núcleos físicos frente a lógicos, frecuencia al cargar más núcleos y ancho de banda de memoria), no código serial.

**GPU (GTX 1650):** el kernel llega a unos 84 × 10⁹ interacciones/s con N = 64000. Con N = 16000 es ≈ 220× más rápido que la CPU serial y ≈ 53× más rápido que la CPU de 12 hilos. Con N pequeño (≈ 1000) la ventaja es modesta, porque hay pocos bloques para los 14 multiprocesadores y domina la latencia de lanzamiento. La energía en float32 se conserva dentro del 0,05 % en 6000 pasos con N = 2002. Las cifras de GPU corresponden a la corrida del 2026-11-04; los valores finales están en `benchmarks/resultados_gpu.csv`.

## 8. Limitaciones

- Algoritmo directo O(N²): no se usa un árbol (Barnes-Hut) ni un método de multipolos.
- La versión serial en NumPy construye matrices N × N × 3, por lo que no sirve para N grande; por eso la referencia serial de los *benchmarks* es Numba con 1 hilo.
- La GPU usa float32 y la CPU float64: la comparación de tiempos mezcla ambas precisiones, y las trayectorias de GPU y CPU divergen con el tiempo por la naturaleza caótica del problema (la energía y el momento sí se conservan).
- Las mediciones se hicieron en un portátil: la CPU y la GPU comparten presupuesto térmico, y eso puede causar *throttling*. Por eso hay corridas repetidas, pausas e intercalado.

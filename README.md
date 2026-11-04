# Simulación de N cuerpos: colisión de galaxias (serial, CPU paralela y GPU)

Proyecto final de Algoritmos y Arquitecturas Computacionales de Alto Rendimiento · Universidad Industrial de Santander · 2026-2

Simulación gravitacional directa de N cuerpos (interacción de todos contra todos, O(N^2)) de la colisión de dos galaxias de disco. Se implementa en tres versiones y se mide el speedup de cada una:

| Versión | Archivo principal | Implementación base | Herramienta | Precisión |
|---|---|---|---|---|
| Serial | serial.py | src/fisica.py | NumPy vectorizado | float64 |
| CPU paralela | paralelo.py --backend cpu | src/cpu_paralelo.py | Numba njit(parallel=True) + prange | float64 |
| GPU | paralelo.py --backend gpu | src/gpu_cuda.py | Numba CUDA (memoria compartida por tiles) | float32 |

https://github.com/user-attachments/assets/97c4cb0d-c3e7-4cc7-8a1d-0b2473a091f9

Integrantes: María José Carreño Gómez y Juan José Mendoza Caballero

---

## 1. Modelo físico

Cada cuerpo i (masa m_i, posición r_i) acelera por la atracción gravitacional de los demás, con suavizado (softening) eps para evitar singularidades numéricas en encuentros cercanos:

a_i = G * sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

La energía total usa el mismo suavizado, asegurando coherencia física y analítica con el campo de fuerzas:

E = K + U,    K = 0.5 * sum m_i |v_i|^2,    U = -0.5 * G * sum_{i} sum_{j != i} m_i * m_j / sqrt(|r_j - r_i|^2 + eps^2)

- Unidades: G = 1; masas y distancias normalizadas (M_0, L_0).
- Parámetros base: eps = 0.05, dt = 0.005 (t = 30 con 6000 pasos).
- Integrador numérico: Leapfrog Kick-Drift-Kick (orden 2, simpléctico y temporalmente reversible).
- Condiciones iniciales (src/condiciones.py):
  - Galaxias de disco tipo Kuzmin/exponencial (núcleo masivo central M_nuc = 0.8, disco estelar M_disco = 0.2).
  - Velocidades circulares equilibradas con la masa encerrada y dispersión del 3% para estabilidad física.
  - Encuentro parabólico fuera de eje (distancia inicial d = 12, parámetro de impacto b = 3, inclinación relativa de 60°). Momento lineal total nulo (P = 0).
- Frontera: Espacio abierto (sin condiciones de frontera periódicas).
- Validación: Comprobada convergencia de orden 2 en órbita kepleriana excéntrica, reversibilidad temporal (|Delta r| ~ 10^-14) y conservación de energía (|Delta E / E_0| < 0.05%, cumpliendo el criterio del 1%).

---

## 2. Estrategia de paralelización

El cuello de botella computacional es el cálculo de interacciones gravitatorias (O(N^2) por paso temporal); la integración de posiciones y velocidades es O(N).

1. CPU paralela (src/cpu_paralelo.py):
   - Paralelización con Numba mediante @njit(parallel=True, fastmath=True) y bucle prange sobre la partícula i.
   - Ausencia de condiciones de carrera: Cada hilo escribe exclusivamente en su fila acc[i, :], evitando secciones críticas o bloqueos atómicos.
   - Control exacto de concurrencia mediante numba.set_num_threads.
2. GPU CUDA (src/gpu_cuda.py):
   - Un hilo por partícula i.
   - Tiling en memoria compartida (Shared Memory): Los hilos cargan cooperativamente bloques de partículas a la memoria compartida local, reduciendo el tráfico al bus de memoria global en un factor igual al tamaño del bloque (TPB).
   - Empaquetado y coalescencia: Arreglo contiguo de 4 componentes float32 [x, y, z, m] para transferencias PCIe alineadas.
   - Datos residentes en memoria de dispositivo (VRAM) entre pasos temporales para minimizar transferencias Host-Device.
3. Verificación Serial == Paralelo:
   - Error relativo máximo de aceleraciones ~ 10^-16 en CPU (límite de precisión doble float64).
   - Error relativo máximo de aceleraciones en GPU (float32) frente a Numba serial float64: de ~9 x 10^-7 (N = 250) a ~7 x 10^-6 (N = 16000), con tolerancia 10^-5; el error crece aproximadamente como la raíz de N. Está en `benchmarks/verificacion_gpu.csv` (`python pruebas/benchmark_gpu.py --solo-verificacion` lo regenera sin repetir el benchmark), que también guarda la dinámica de 400 pasos (diferencia máxima de posición ~1,2 x 10^-4) y la conservación de energía en float32 (|ΔE/E| máx. ~5 x 10^-4 en 6000 pasos, N = 2002). La tolerancia de 10^-5 se verificó hasta N = 16000; por la tendencia, no debe extrapolarse a N = 64000 sin comprobarlo.

---

## 3. Hardware y software usados

- CPU: Intel Core i5-11400H @ 2.70 GHz (6 núcleos físicos / 12 hilos lógicos)
- GPU: NVIDIA GeForce GTX 1650 Mobile (4 GiB GDDR6, 14 SMs, Compute Capability 7.5)
- Sistema Operativo: Windows 11 (build 10.0.26300)
- Entorno Python: Python 3.11 en Conda (nbody-gpu)
- Librerías (`requirements.txt`): NumPy 2.4.6, Numba 0.66.0, Numba-CUDA 0.30.4, Matplotlib 3.11.2
- Versión de Numba con la que se hizo cada medición (la registra el propio script en `resultados/hardware*.txt`):

| Mediciones | Fecha | Numba | Fuente |
|---|---|---|---|
| CPU (`benchmark_cpu.py`, corridas c1 a c3) | 2026-10-30 | 0.68.0 | `resultados/hardware.txt` |
| GPU (`benchmark_gpu.py`, dos corridas) y verificación GPU | 2026-11-04 | 0.66.0 | `resultados/hardware_gpu.txt` |

  Las mediciones de CPU y de GPU se hicieron en momentos distintos y con versiones distintas de Numba. Los resultados de CPU no dependen de la GPU y viceversa, pero para reproducirlos con exactitud conviene usar la versión indicada en cada fila. Entorno exacto: `conda list --export > entorno_conda.txt`.
- CUDA Runtime: 12.9

---

## 4. Estructura del repositorio

- serial.py: Punto de entrada de simulación serial
- paralelo.py: Punto de entrada de simulación paralela (CPU / GPU)
- medir.py: Script maestro de benchmarking (genera CSVs y figuras)
- requirements.txt: Dependencias del proyecto
- README.md: Documentación completa e instrucciones
- src/: Módulos del modelo (fisica.py, cpu_paralelo.py, gpu_cuda.py, condiciones.py)
- pruebas/: Scripts de validación física, benchmarks y renderizado
- benchmarks/: Datos crudos en CSV de las mediciones de rendimiento
- resultados/: Gráficas generadas, tablas consolidadas y diagnósticos

---

## 5. Instrucciones de ejecución y reproducción

Activar previamente el entorno Conda:
conda activate nbody-gpu

### A. Ejecución de simulaciones directas
- Versión serial:
  python serial.py --n 500 --pasos 400
- Versión paralela en CPU:
  python paralelo.py --backend cpu --hilos 6 --n 1000 --pasos 600
- Versión paralela en GPU:
  python paralelo.py --backend gpu --n 4000 --pasos 1000

### B. Medición de desempeño y generación de CSVs (Script medir.py)
- Medición completa (CPU + GPU + Análisis de Amdahl):
  python medir.py --all
- Medición exclusiva de CPU (3 corridas con pausas térmicas):
  python medir.py --cpu
- Medición exclusiva de GPU:
  python medir.py --gpu
- Prueba rápida (para verificar ejecución sin esperas prolongadas):
  python medir.py --all --rapido
- Reanalizar CSVs existentes:
  python medir.py --analizar
- Speedup frente a cada referencia (NumPy, Numba 1 hilo, Numba paralelo):
  python pruebas/baselines_cpu.py

### C. Validación física y visualización
- Validación de integrador (convergencia de orden 2 y dt):
  python pruebas/validacion_fisica.py
- Generar video/animación de la colisión (MP4/GIF):
  python pruebas/visualizar.py --backend gpu --n 4000 --pasos 6000

---

## 6. Resultados y síntesis de desempeño

### Referencias del speedup (qué es T1)

El speedup depende de la referencia con que se compare. En este proyecto hay tres, y no deben mezclarse:

| Ref. | Qué es | Dónde está |
|---|---|---|
| A | NumPy vectorizado, es decir el código de `serial.py` | `numpy` en `resultados_cpu*.csv` (medido solo hasta N = 2000) |
| B | Numba compilado a código máquina, 1 hilo, sin `prange` | `numba_1_hilo` (CPU) y `cpu_serial` (GPU) |
| C | El código paralelo ejecutado con p hilos (T1 es C con p = 1) | `numba_paralelo` y `numba_paralelo_hilos` |

- **Speedup de paralelismo (S = T1/Tp, criterio 5):** usa T1 = C con 1 hilo, medido en la misma ronda que Tp. Es la tabla `resultados/tabla_speedup_cpu.csv` y la figura `analisis_cpu.png`. B y C con 1 hilo son prácticamente el mismo código.
- **Frente a `serial.py`:** como `serial.py` es NumPy (A), la ganancia total frente a él mezcla dos efectos distintos: compilar a código máquina y paralelizar.

Resultados para 12 hilos, mediana de 4 corridas (`python pruebas/baselines_cpu.py` los regenera en `resultados/tabla_baselines_cpu.csv`):

| N | A: NumPy (ms) | B: Numba 1 h (ms) | C: Numba 12 h (ms) | A/B (compilación) | B/C (paralelismo) | A/C (total) |
|---|---|---|---|---|---|---|
| 250 | 4,93 | 0,489 | 0,082 | 13,2 | 4,5 | 74,9 |
| 500 | 20,02 | 0,991 | 0,235 | 20,1 | 4,3 | 85,1 |
| 1000 | 53,29 | 4,067 | 0,753 | 15,1 | 5,6 | 83,1 |
| 2000 | 192,50 | 10,108 | 2,340 | 19,2 | 4,4 | 86,3 |
| 4000 | no medido | 40,170 | 8,701 | - | 4,7 | - |
| 8000 | no medido | 162,593 | 37,351 | - | 4,4 | - |
| 16000 | no medido | 653,185 | 155,011 | - | 4,1 | - |

Frente a `serial.py` la versión paralela es unas 85 veces más rápida (N = 2000), pero solo ~4,4x corresponde a paralelismo; el resto (~19x) viene de compilar con Numba. Por eso el análisis de eficiencia y de Amdahl usa B/C y no A/C. En esta tabla N es el número **total** de cuerpos; en `serial.py` y `paralelo.py`, `--n` es el número por galaxia.

### Protocolo de medición (repeticiones y dispersión)

- Todos los tiempos se miden con `time.perf_counter` y la primera llamada (compilación JIT) se descarta.
- Una **corrida** es una ejecución completa e independiente de `benchmark_cpu.py`; se hicieron 4 (`resultados_cpu.csv` y `_c1`, `_c2`, `_c3`, con pausas de enfriamiento). Una **repetición** es una medición dentro de una corrida. La columna `n_corridas` de `tabla_speedup_cpu.csv` cuenta corridas, no repeticiones.
- Se reporta la mediana de cada corrida, luego la mediana entre corridas y el rango mínimo-máximo entre corridas (`analizar_cpu.py`).

| Bloque | Repeticiones por corrida | Corridas | Muestras por configuración |
|---|---|---|---|
| CPU: escalado por hilos (speedup, eficiencia, Amdahl) | 15 | 4 | 60 |
| CPU: tiempo vs N, Numba 1 hilo y paralelo (refs. B y C) | 7 | 4 | 28 |
| CPU: NumPy, `serial.py` (ref. A) | 3 | 4 | 12 |
| GPU: kernel y transferencias | 15 | 1 | 15 |
| GPU: referencias de CPU (serial y paralelo) | 5 | 1 | 5 |

`python pruebas/resumen_repeticiones.py` regenera esta tabla desde los CSV (`resultados/tabla_repeticiones.csv`).

Dispersión: con N = 16000 y 12 hilos, la diferencia entre el mínimo y el máximo dentro de una corrida llega al 14 %-21 % de la mediana, y las medianas de las 4 corridas van de 152 a 173 ms. Por eso los speedups se reportan con su rango y no como un único valor.

Limitaciones: (1) NumPy se repitió solo 3 veces por corrida (por debajo de las 5 recomendadas), así que los speedups frente a `serial.py` (ref. A) son menos robustos que los demás; (2) las mediciones de GPU provienen de una sola corrida, sin dispersión entre corridas.

1. CPU multihilo (Numba):
   - Speedup máximo obtenido de ~4.4x con 12 hilos, frente al mismo código paralelo ejecutado con 1 hilo (referencia C de la tabla anterior).
   - Eficiencia paralela de ~73% a 4 hilos, descendiendo a ~37% a 12 hilos.
   - Ajuste por mínimos cuadrados de la ley de Amdahl arroja una fracción paralelizable efectiva f ~ 0.84. Esta cota está limitada por el factor de contención de memoria compartida L3 y la contención en Hyper-Threading (6 núcleos físicos vs 12 lógicos).
2. GPU CUDA (GTX 1650):
   - El kernel de GPU alcanza entre ~65 x 10^9 y ~84 x 10^9 interacciones gravitatorias/segundo para N = 64000, según la corrida (dos corridas independientes, ver más abajo).
   - Con N = 16000, el kernel en GPU es entre ~220x y ~270x más rápido que Numba compilado con 1 hilo (referencia B) y entre ~50x y ~53x más rápido que Numba CPU con 12 hilos, según la corrida. La variación entre corridas (la CPU serial tardó 0,71 s en una y 1,24 s en otra con N = 16000) es típica de un portátil con limitación térmica; por eso se reporta el rango y no un solo valor. Estas comparaciones usan la referencia B, no `serial.py` (NumPy).
   - **Precisión distinta (leer antes de interpretar el speedup de GPU):** el kernel de GPU trabaja en float32 y todas las referencias de CPU en float64, así que estas razones de tiempo no comparan exactamente el mismo trabajo numérico. La diferencia favorece a la GPU: en tarjetas de consumo como la GTX 1650 (arquitectura Turing) la doble precisión es mucho más lenta que la simple (del orden de 32 veces menos rendimiento, según la especificación de la arquitectura), de modo que una GPU en float64 daría un speedup bastante menor. En la CPU, float32 también permitiría procesar el doble de datos por instrucción vectorial. Por eso el resultado debe leerse como «GPU en float32 frente a CPU en float64»; la CPU en float32 no se midió.
   - **Costo de usar float32:** el error relativo de las aceleraciones es de ~10^-6 a ~7 x 10^-6 según N (frente a ~10^-16 en CPU) y, como la dinámica es caótica, las trayectorias de GPU y CPU divergen con el tiempo aunque cada paso sea correcto. La conservación de energía en float32 queda registrada en `benchmarks/verificacion_gpu.csv` y cumple el criterio del 1 %.
   - Costo de transferencia PCIe (H2D + D2H) como fracción del tiempo total (kernel + transferencias). Dos corridas de GPU (`resultados_gpu_corrida1.csv` y `resultados_gpu.csv`, la más reciente), rango entre ellas: N = 1000, 50-56 %; N = 4000, 38-59 %; N = 8000, 17-37 %; N = 16000, 7-18 %; N = 32000, 2,5-6 %; N = 64000, 1-2 %. Solo baja del 5 % de forma consistente en N = 64000; en N = 32000 depende de la corrida. Los tiempos de kernel suponen que posiciones y velocidades permanecen en la GPU entre pasos, por lo que en la simulación completa las transferencias no se pagan en cada paso.
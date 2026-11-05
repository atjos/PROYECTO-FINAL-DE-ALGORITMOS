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
- Validación (`pruebas/validacion_fisica.py` y `serial.py`):
  - Convergencia de orden 2 en órbita kepleriana excéntrica: al dividir dt entre 2 el error se reduce en un factor de ~3,95 a 3,98 (esperado: 4).
  - Reversibilidad temporal: |Delta r| ~ 2 x 10^-13 tras ir y volver.
  - Conservación de energía en la colisión completa (N = 302, dt = 0.005, t = 30): |Delta E / E_0| máximo de 4,8 x 10^-4 (0,05 %) con la semilla 1 de la validación y de 6,1 x 10^-4 (0,06 %) con la semilla 42 de `serial.py`. En ambos casos queda por debajo de 0,1 %, muy por debajo del criterio del 1 %. Con dt = 0.0025 el máximo fue 8,2 x 10^-4: en esta colisión el error de energía no decrece de forma monótona al reducir dt, por lo que dt = 0.005 es el valor base.

---

## 2. Estrategia de paralelización

El cuello de botella computacional es el cálculo de interacciones gravitatorias (O(N^2) por paso temporal); la integración de posiciones y velocidades es O(N).

Esto se mide, no solo se afirma: `pruebas/perfil_serial.py` (`python medir.py --perfil`) cronometra con `time.perf_counter`, por separado, las fuerzas (O(N^2) cada paso), la energía (O(N^2), solo cada `cada` pasos) y el resto del paso (kick-drift-kick, momentos y copia; O(N)). Guarda `resultados/tabla_perfil.csv` con la fracción del paso que corresponde a cada bloque y la fracción `f_medida` = (fuerzas + energía/cada) / paso, que es la fracción paralelizable real del código. Esa `f_medida` se contrasta con la f efectiva del ajuste de Amdahl (sección 6).

**Perfil del paso de simulación** (`resultados/tabla_perfil.csv`, mediana de 9 repeticiones; la energía se registra cada 10 pasos). Los tiempos son de una sesión y las fracciones son razones dentro de esa misma sesión; el tiempo absoluto de una sesión a otra puede variar hasta ~2x en este portátil.

| Versión | N | Fuerzas (ms) | Energía (ms) | Resto O(N) (ms) | % fuerzas | % energía | % resto | f medida |
|---|---|---|---|---|---|---|---|---|
| NumPy (`serial.py`) | 250 | 7,527 | 3,529 | 0,018 | 95,30 | 4,47 | 0,231 | 0,9977 |
| NumPy (`serial.py`) | 500 | 20,114 | 13,792 | 0,024 | 93,48 | 6,41 | 0,113 | 0,9989 |
| NumPy (`serial.py`) | 1000 | 84,422 | 70,497 | 0,021 | 92,27 | 7,71 | 0,023 | 0,9998 |
| NumPy (`serial.py`) | 2000 | 339,410 | 261,495 | 0,051 | 92,83 | 7,15 | 0,014 | 0,9999 |
| Numba 1 hilo (ref. B) | 250 | 0,213 | 0,180 | 0,017 | 85,93 | 7,27 | 6,795 | 0,9321 |
| Numba 1 hilo (ref. B) | 500 | 1,436 | 0,419 | 0,026 | 95,52 | 2,79 | 1,697 | 0,9830 |
| Numba 1 hilo (ref. B) | 1000 | 5,574 | 0,862 | 0,032 | 97,92 | 1,51 | 0,568 | 0,9943 |
| Numba 1 hilo (ref. B) | 2000 | 17,433 | 3,215 | 0,030 | 98,02 | 1,81 | 0,169 | 0,9983 |
| Numba 1 hilo (ref. B) | 4000 | 69,680 | 15,396 | 0,062 | 97,75 | 2,16 | 0,087 | 0,9991 |
| Numba 1 hilo (ref. B) | 8000 | 291,868 | 54,016 | 0,109 | 98,15 | 1,82 | 0,037 | 0,9996 |
| Numba 1 hilo (ref. B) | 16000 | 1121,750 | 211,249 | 0,629 | 98,10 | 1,85 | 0,055 | 0,9995 |

Las fuerzas son entre 86 % y 98 % del paso (≈98 % en Numba para N ≥ 2000), lo que confirma que son la parte costosa y la que se paraleliza. La energía también es O(N²) y va en paralelo en la versión Numba; el resto del paso (O(N)) es lo único que queda serial. `f medida` = (fuerzas + energía/10) / paso es la fracción paralelizable real del código.

Cota de Amdahl con esa `f medida` frente al speedup medido (referencia C, mediana de 4 corridas):

| N | f medida | Cota a 6 hilos | S medido a 6 | Cota a 12 hilos | S medido a 12 |
|---|---|---|---|---|---|
| 2000 | 0,9983 | 5,95 | 3,43 | 11,8 | 4,35 |
| 16000 | 0,9995 | 5,98 | 3,00 | 11,9 | 4,40 |

Con N ≥ 2000 la fracción serial del código (< 0,2 %) permitiría casi el speedup ideal, y se obtiene menos de la mitad. La pérdida no viene de una parte serial del código. La f ≈ 0,84 del ajuste de Amdahl es una fracción *efectiva*: absorbe lo que limita el hardware (frecuencia con todos los núcleos activos, límites térmicos del portátil, hilos lógicos que comparten unidades de ejecución). Solo en N pequeño (N = 250, `f medida` = 0,93) la parte serial pesa: la cota a 12 hilos es ≈ 6,9, frente a 4,5 medido.

1. CPU paralela (src/cpu_paralelo.py):
   - Paralelización con Numba mediante @njit(parallel=True, fastmath=True) y bucle prange sobre la partícula i.
   - Ausencia de condiciones de carrera: Cada hilo escribe exclusivamente en su fila acc[i, :], evitando secciones críticas o bloqueos atómicos.
   - Control exacto de concurrencia mediante numba.set_num_threads.
   - GIL: `prange` de Numba compila el bucle a código máquina y se ejecuta fuera del intérprete de Python, así que los hilos corren en paralelo real (con `threading` sobre Python puro el GIL lo impediría).
   - Reducción: en `_energia_potencial`, `U += ...` dentro de `prange` es una reducción escalar que Numba reconoce y resuelve con una copia privada por hilo, combinadas al final; no hay escritura concurrente sobre una misma variable. El orden de suma cambia con el número de hilos, por eso U puede diferir en el último dígito entre ejecuciones.
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
- medir.py: Script maestro; regenera todos los CSV, tablas y figuras (ver sección 5)
- requirements.txt: Dependencias del proyecto
- README.md: Documentación completa e instrucciones
- src/: Módulos del modelo (fisica.py, cpu_paralelo.py, gpu_cuda.py, condiciones.py)
- pruebas/: Scripts de validación física, benchmarks y renderizado
  - test_entorno.py: comprueba que Numba y CUDA funcionan en el entorno
  - prueba_serial.py: prueba rápida de la versión serial
  - prueba_condiciones.py: condiciones iniciales y estabilidad de una galaxia aislada
  - validacion_fisica.py: orden del integrador, reversibilidad y efecto de dt
  - perfil_serial.py: perfilado del paso de simulacion (fuerzas vs energia vs resto) y fraccion paralelizable medida
  - benchmark_cpu.py / benchmark_gpu.py: mediciones de tiempo y verificación serial vs paralelo
  - analizar_cpu.py / baselines_cpu.py / resumen_repeticiones.py: tablas y gráficas derivadas de los CSV
  - visualizar.py: render de la colision (imagen final y video)
- benchmarks/: Datos crudos en CSV de las mediciones de rendimiento
- resultados/: Gráficas generadas, tablas consolidadas y diagnósticos
- SIMULACIÓN WEB/: versión interactiva en el navegador (HTML autocontenido)

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

`paralelo.py` hace un calentamiento antes de cronometrar (compilación JIT en CPU, arranque de CUDA en GPU) con un problema mínimo y reporta su duración aparte; el tiempo de la simulación no incluye la compilación. Ese tiempo sí incluye los registros periódicos de energía y momentos. Las mediciones formales de speedup están en `medir.py` (sección B), no en estos tiempos.

### B. Regenerar todo con medir.py

`medir.py` es el punto único para reproducir los CSV, las tablas y las figuras. Al terminar imprime un resumen con `[OK]` o `[FALLO]` por paso.

| Comando | Qué hace |
|---|---|
| `python medir.py --all` | Todo: CPU, GPU, tablas y figuras (incluye el video) |
| `python medir.py --all --rapido` | Lo mismo con tamaños reducidos; sirve para comprobar que todo corre. Las mediciones van a `benchmarks/rapido/` y `resultados/rapido/`, así que no tocan los resultados publicados |
| `python medir.py --cpu` | Mediciones de CPU (4 corridas con pausa de 20 s para enfriar) y reanálisis de tablas |
| `python medir.py --gpu` | Mediciones de GPU (requiere CUDA) y reanálisis de tablas |
| `python medir.py --perfil` | Perfilado del paso de simulación (`resultados/tabla_perfil.csv`). También lo ejecutan `--cpu` y `--all` |
| `python medir.py --analizar` | Solo reanaliza los CSV existentes: speedup, eficiencia, Amdahl y tablas |
| `python medir.py --figuras` | Figuras de física y de la colisión, incluido el video |
| `python medir.py --figuras --sin-video` | Igual, sin renderizar el video (mucho más rápido) |

Opciones de `--cpu`: `--corridas N` y `--pausa SEGUNDOS`.

Antes de medir (`--cpu`, `--gpu`, `--all`), `medir.py` compara las versiones instaladas con las de `requirements.txt` y muestra un aviso si no coinciden, para que ninguna medición quede hecha con versiones distintas de las documentadas. El aviso no detiene la ejecución.

### C. Qué comando genera cada archivo de resultados

Cada fila se puede ejecutar por separado (columna "Script") o, más cómodo, con `medir.py`.

| Archivo | Script | Con medir.py |
|---|---|---|
| `benchmarks/resultados_cpu_*_c*.csv`, `benchmarks/verificacion_cpu.csv`, `resultados/benchmark_cpu.png` (figura de la última corrida), `resultados/hardware.txt` | `python pruebas/benchmark_cpu.py` | `--cpu` |
| `benchmarks/resultados_gpu.csv`, `benchmarks/verificacion_gpu.csv`, `resultados/benchmark_gpu.png`, `resultados/hardware_gpu.txt` | `python pruebas/benchmark_gpu.py` (solo la verificación: `--solo-verificacion`) | `--gpu` |
| `resultados/tabla_perfil.csv` (fracción del paso en fuerzas, energía y resto; `f_medida`) | `python pruebas/perfil_serial.py` | `--perfil` (también `--cpu`, `--all`) |
| `resultados/analisis_cpu.png` (speedup y eficiencia con Amdahl), `resultados/tabla_speedup_cpu.csv` | `python pruebas/analizar_cpu.py` | `--analizar` |
| `resultados/tabla_baselines_cpu.csv` (speedup frente a cada referencia) | `python pruebas/baselines_cpu.py` | `--analizar` |
| `resultados/tabla_repeticiones.csv` | `python pruebas/resumen_repeticiones.py` | `--analizar` |
| `resultados/condiciones_iniciales.png`, `resultados/estabilidad_galaxia.png` | `python pruebas/prueba_condiciones.py` | `--figuras` |
| `resultados/validacion_dt.png` (orden 2, reversibilidad, efecto de dt) | `python pruebas/validacion_fisica.py` (tarda unos minutos) | `--figuras` |
| `resultados/colision_final.png`, `resultados/colision.mp4` | `python pruebas/visualizar.py --backend gpu --n 4000 --pasos 6000` (con `--rapido` para una prueba corta) | `--figuras` |

Notas:
- `visualizar.py` usa la GPU si hay CUDA (`--backend auto`); con `--backend cpu` o `numpy` corre sin GPU, más lento. Para guardar MP4 necesita ffmpeg; sin él, genera un GIF si está `imageio`.
- Los scripts de `pruebas/` se pueden ejecutar desde cualquier carpeta.
- Cada ejecución de `--cpu` escribe un lote nuevo de CSV (`resultados_cpu_<fecha>_c<k>.csv`). Al terminar todas las corridas, los CSV de sesiones anteriores se mueven (no se borran) a `benchmarks/archivo/<fecha>/`, de modo que `benchmarks/` contiene siempre un solo lote y el análisis, que lee todos los `resultados_cpu*.csv` de esa carpeta, no mezcla mediciones de sesiones distintas. Si una ejecución falla a medio camino, los datos anteriores siguen en su lugar.
- Las pruebas con `--rapido` nunca sobrescriben los resultados publicados: escriben en `benchmarks/rapido/` y `resultados/rapido/`, y el análisis las ignora.
- Las simulaciones sueltas y la comprobación del entorno: `python pruebas/test_entorno.py` y `python pruebas/prueba_serial.py`.

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
- Una **corrida** es una ejecución completa e independiente de `benchmark_cpu.py`; se hicieron 4 con pausas de enfriamiento. Las cuatro publicadas están en `benchmarks/` como `resultados_cpu.csv` y `resultados_cpu_20261030_005902_c1.csv` a `_c3.csv`; una sesión nueva (`python medir.py --cpu`) produce también 4, siempre con el formato `resultados_cpu_<fecha>_c1.csv` a `_c4.csv`. Una **repetición** es una medición dentro de una corrida. La columna `n_corridas` de `tabla_speedup_cpu.csv` cuenta corridas, no repeticiones.
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


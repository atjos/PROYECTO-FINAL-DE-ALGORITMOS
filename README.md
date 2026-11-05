<div align="center">

# Simulación de N cuerpos: colisión de galaxias

### Versiones serial, CPU paralela (Numba) y GPU (CUDA)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-0.66.0-00A3E0?logo=numba&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.9-76B900?logo=nvidia&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.6-013243?logo=numpy&logoColor=white)

**Proyecto final de Algoritmos y Arquitecturas Computacionales de Alto Rendimiento · Universidad Industrial de Santander · 2026-2**

**Integrantes:** María José Carreño Gómez · Juan José Mendoza Caballero

</div>

---

Simulación gravitacional directa de $N$ cuerpos (interacción de todos contra todos, complejidad $\mathcal{O}(N^2)$ por paso) de la colisión de dos galaxias de disco. Se implementa en tres versiones y se mide el speedup de cada una:

| Versión | Archivo principal | Implementación base | Herramienta | Precisión |
|:---|:---|:---|:---|:---:|
| **Serial** | `serial.py` | `src/fisica.py` | NumPy vectorizado | `float64` |
| **CPU paralela** | `paralelo.py --backend cpu` | `src/cpu_paralelo.py` | Numba `njit(parallel=True)` + `prange` | `float64` |
| **GPU** | `paralelo.py --backend gpu` | `src/gpu_cuda.py` | Numba CUDA (memoria compartida por *tiles*) | `float32` |

https://github.com/user-attachments/assets/97c4cb0d-c3e7-4cc7-8a1d-0b2473a091f9

## Contenido

1. [Modelo físico](#1-modelo-físico)
2. [Estrategia de paralelización](#2-estrategia-de-paralelización)
3. [Hardware y software usados](#3-hardware-y-software-usados)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Instrucciones de ejecución y reproducción](#5-instrucciones-de-ejecución-y-reproducción)
6. [Resultados y síntesis de desempeño](#6-resultados-y-síntesis-de-desempeño)

---

## 1. Modelo físico

Cada cuerpo $i$ (masa $m_i$, posición $\mathbf{r}_i$) acelera por la atracción gravitacional de los demás, con un suavizado (*softening*) $\varepsilon$ que evita singularidades numéricas en encuentros cercanos:

```math
\mathbf{a}_i \;=\; G \sum_{j \neq i} \frac{m_j \,\left(\mathbf{r}_j - \mathbf{r}_i\right)}{\left(\lVert \mathbf{r}_j - \mathbf{r}_i \rVert^{2} + \varepsilon^{2}\right)^{3/2}}
```

La energía total usa el mismo suavizado, lo que garantiza coherencia física y analítica con el campo de fuerzas:

```math
E = K + U, \qquad
K = \frac{1}{2}\sum_i m_i \,\lVert \mathbf{v}_i \rVert^{2}, \qquad
U = -\frac{G}{2} \sum_i \sum_{j \neq i} \frac{m_i\, m_j}{\sqrt{\lVert \mathbf{r}_j - \mathbf{r}_i \rVert^{2} + \varepsilon^{2}}}
```

### Integrador: Leapfrog *Kick-Drift-Kick* (orden 2)

```math
\begin{aligned}
\mathbf{v}_{n+1/2} &= \mathbf{v}_n + \tfrac{\Delta t}{2}\,\mathbf{a}(\mathbf{r}_n) \\
\mathbf{r}_{n+1} &= \mathbf{r}_n + \Delta t \;\mathbf{v}_{n+1/2} \\
\mathbf{v}_{n+1} &= \mathbf{v}_{n+1/2} + \tfrac{\Delta t}{2}\,\mathbf{a}(\mathbf{r}_{n+1})
\end{aligned}
```

Es simpléctico y temporalmente reversible.

### Parámetros y condiciones

| Aspecto | Valor |
|:---|:---|
| Unidades | $G = 1$; masas y distancias normalizadas ($M_0$, $L_0$) |
| Suavizado | $\varepsilon = 0{,}05$ |
| Paso de tiempo | $\Delta t = 0{,}005$ ($t = 30$ con 6000 pasos) |
| Galaxias | Disco tipo Kuzmin/exponencial: núcleo masivo $M_{\text{nuc}} = 0{,}8$ y disco estelar $M_{\text{disco}} = 0{,}2$ |
| Velocidades | Circulares, equilibradas con la masa encerrada, con dispersión del 3 % para estabilidad física |
| Encuentro | Parabólico fuera de eje: distancia inicial $d = 12$, parámetro de impacto $b = 3$, inclinación relativa de $60^\circ$ |
| Momento lineal total | $\mathbf{P} = \mathbf{0}$ |
| Frontera | Espacio abierto (sin condiciones periódicas) |
| Condiciones iniciales | `src/condiciones.py` |

### Validación (`pruebas/validacion_fisica.py` y `serial.py`)

| Prueba | Resultado | Criterio |
|:---|:---|:---:|
| Convergencia en órbita kepleriana excéntrica ($e = 0{,}6$) | Al dividir $\Delta t$ entre 2, el error se reduce en un factor de $3{,}95$ a $3{,}98$ | $\approx 4$ (orden 2) ✅ |
| Reversibilidad temporal | $\lVert \Delta \mathbf{r} \rVert \sim 2{,}8 \times 10^{-12}$ tras ir y volver | $\sim 10^{-12}$ |
| Energía, colisión completa ($N = 302$, $\Delta t = 0{,}005$, $t = 30$), semilla 1 | $\max \lvert \Delta E / E_0 \rvert = 4{,}6 \times 10^{-4}$ (0,046 %) | $`< 1\,\%`$ ✅ |
| Energía, colisión completa, semilla 42 (`serial.py`) | $\max \lvert \Delta E / E_0 \rvert = 6{,}1 \times 10^{-4}$ (0,06 %) | $`< 1\,\%`$ ✅ |

> [!NOTE]
> Con $\Delta t = 0{,}0025$ el máximo fue $8{,}2 \times 10^{-4}$: en esta colisión el error de energía no decrece de forma monótona al reducir $\Delta t$, por lo que $\Delta t = 0{,}005$ es el valor base.

---

## 2. Estrategia de paralelización

El cuello de botella es el cálculo de interacciones gravitatorias, $\mathcal{O}(N^2)$ por paso; la integración de posiciones y velocidades es $\mathcal{O}(N)$.

Esto se **mide**, no solo se afirma: `pruebas/perfil_serial.py` (`python medir.py --perfil`) cronometra con `time.perf_counter`, por separado, las fuerzas ($\mathcal{O}(N^2)$ en cada paso), la energía ($\mathcal{O}(N^2)$, solo cada $c$ pasos) y el resto del paso (*kick-drift-kick*, momentos y copia; $\mathcal{O}(N)$). Guarda `resultados/tabla_perfil.csv` con la fracción del paso de cada bloque y la **fracción paralelizable medida**:

```math
f_{\text{medida}} \;=\; \frac{t_{\text{fuerzas}} + t_{\text{energía}}/c}{t_{\text{paso}}}
```

Esta $f_{\text{medida}}$ se contrasta con la $f$ efectiva del ajuste de Amdahl (sección 6).

### Perfil del paso de simulación

> Mediana de 9 repeticiones; la energía se registra cada $c = 10$ pasos (`resultados/tabla_perfil.csv`). Los tiempos son de una sesión y las fracciones son razones dentro de esa misma sesión; el tiempo absoluto de una sesión a otra puede variar hasta ~2× en este portátil.

| Versión | $N$ | Fuerzas (ms) | Energía (ms) | Resto $\mathcal{O}(N)$ (ms) | % fuerzas | % energía | % resto | $f_{\text{medida}}$ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| NumPy (`serial.py`) | 250 | 3,318 | 2,339 | 0,011 | 93,13 | 6,57 | 0,299 | 0,9970 |
| NumPy (`serial.py`) | 500 | 14,445 | 10,870 | 0,012 | 92,93 | 6,99 | 0,079 | 0,9992 |
| NumPy (`serial.py`) | 1000 | 51,296 | 42,558 | 0,017 | 92,31 | 7,66 | 0,031 | 0,9997 |
| NumPy (`serial.py`) | 2000 | 196,436 | 151,115 | 0,024 | 92,85 | 7,14 | 0,011 | 0,9999 |
| Numba 1 hilo (ref. B) | 250 | 0,170 | 0,119 | 0,013 | 87,32 | 6,08 | 6,595 | 0,9340 |
| Numba 1 hilo (ref. B) | 500 | 0,652 | 0,166 | 0,014 | 95,58 | 2,43 | 1,985 | 0,9801 |
| Numba 1 hilo (ref. B) | 1000 | 2,451 | 0,507 | 0,019 | 97,22 | 2,01 | 0,771 | 0,9923 |
| Numba 1 hilo (ref. B) | 2000 | 10,440 | 1,825 | 0,023 | 98,07 | 1,71 | 0,218 | 0,9978 |
| Numba 1 hilo (ref. B) | 4000 | 42,234 | 8,081 | 0,041 | 98,03 | 1,88 | 0,095 | 0,9991 |
| Numba 1 hilo (ref. B) | 8000 | 171,697 | 36,394 | 0,084 | 97,88 | 2,07 | 0,048 | 0,9995 |
| Numba 1 hilo (ref. B) | 16000 | 685,627 | 147,980 | 0,482 | 97,82 | 2,11 | 0,069 | 0,9993 |

Las fuerzas son entre el 87 % y el 98 % del paso (≈ 98 % en Numba para $N \geq 2000$), lo que confirma que son la parte costosa y la que se paraleliza. La energía también es $\mathcal{O}(N^2)$ y va en paralelo en la versión Numba; el resto del paso ($\mathcal{O}(N)$) es lo único que queda serial.

### Cota de Amdahl con la $f_{\text{medida}}$ frente al speedup medido

Referencia C, mediana de 4 corridas. La cota se obtiene de:

```math
S_p^{\text{cota}} \;=\; \frac{1}{(1 - f_{\text{medida}}) + \dfrac{f_{\text{medida}}}{p}}
```

| $N$ | $f_{\text{medida}}$ | Cota $S_6$ | $S_6$ medido | Cota $S_{12}$ | $S_{12}$ medido |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0,9978 | 5,94 | **4,96** | 11,7 | **5,17** |
| 16000 | 0,9993 | 5,98 | **4,52** | 11,9 | **4,31** |

Con $N \geq 2000$ la fracción serial del código ($`< 0{,}3\,\%`$) permitiría casi el speedup ideal. Con 6 hilos (los 6 núcleos físicos) se alcanza el **84 %** y el **76 %** de esa cota; la pérdida grande aparece al pasar a hilos lógicos: con 12 hilos se obtiene el **44 %** y el **36 %** de la cota.

> [!IMPORTANT]
> La pérdida **no** viene de una parte serial del código. La $f \approx 0{,}88$ del ajuste de Amdahl es una fracción *efectiva*: absorbe lo que limita el hardware (frecuencia con todos los núcleos activos, límites térmicos del portátil, hilos lógicos que comparten unidades de ejecución). Solo con $N$ pequeño ($N = 250$, $f_{\text{medida}} = 0{,}93$) la parte serial pesa: la cota a 12 hilos es $\approx 7{,}0$, frente a $3{,}7$ medido.

### Implementación

####  CPU paralela (`src/cpu_paralelo.py`)

- **Paralelización:** Numba con `@njit(parallel=True, fastmath=True)` y bucle `prange` sobre la partícula $i$.
- **Sin condiciones de carrera:** cada hilo escribe exclusivamente en su fila `acc[i, :]`, evitando secciones críticas o bloqueos atómicos.
- **Control de concurrencia:** `numba.set_num_threads`.
- **GIL:** `prange` compila el bucle a código máquina y se ejecuta fuera del intérprete de Python, así que los hilos corren en paralelo real (con `threading` sobre Python puro el GIL lo impediría).
- **Reducción:** en `_energia_potencial`, `U += ...` dentro de `prange` es una reducción escalar que Numba reconoce y resuelve con una copia privada por hilo, combinadas al final; no hay escritura concurrente sobre una misma variable. El orden de suma cambia con el número de hilos, por eso $U$ puede diferir en el último dígito entre ejecuciones.

####  GPU CUDA (`src/gpu_cuda.py`)

- **Un hilo por partícula** $i$.
- **Tiling en memoria compartida:** los hilos cargan cooperativamente bloques de partículas a la memoria compartida, reduciendo el tráfico a memoria global en un factor igual al tamaño del bloque (TPB).
- **Empaquetado y coalescencia:** arreglo contiguo de 4 componentes `float32` `[x, y, z, m]` para transferencias PCIe alineadas.
- **Datos residentes en VRAM** entre pasos temporales, para minimizar transferencias Host-Device.

#### ✅ Verificación serial ≡ paralelo

| Comparación | Magnitud | Valor | Tolerancia |
|:---|:---|:---:|:---:|
| CPU paralela vs referencia (`float64`) | Error relativo máx. de aceleraciones | $\sim 10^{-16}$ | $10^{-10}$ |
| CPU, dinámica de 400 pasos ($N = 302$) | Diferencia máx. de posición | $4{,}1 \times 10^{-14}$ | — |
| GPU (`float32`) vs Numba serial (`float64`) | Error relativo máx. de aceleraciones | $9 \times 10^{-7}$ ($N = 250$) a $7 \times 10^{-6}$ ($N = 16000$) | $10^{-5}$ |
| GPU, dinámica de 400 pasos | Diferencia máx. de posición | $\sim 1{,}2 \times 10^{-4}$ | informativo |
| GPU, conservación de energía en `float32` ($N = 2002$, 6000 pasos) | $\max \lvert \Delta E/E \rvert$ | $\sim 5 \times 10^{-4}$ | $`1\,\%`$ |

El error de la GPU crece aproximadamente como $\sqrt{N}$. Los datos están en `benchmarks/verificacion_gpu.csv` (`python pruebas/benchmark_gpu.py --solo-verificacion` lo regenera sin repetir el benchmark).

> [!WARNING]
> La tolerancia de $10^{-5}$ se verificó hasta $N = 16000$; por la tendencia, no debe extrapolarse a $N = 64000$ sin comprobarlo.

---

## 3. Hardware y software usados

| Componente | Detalle |
|:---|:---|
| **CPU** | Intel Core i5-11400H @ 2,70 GHz — 6 núcleos físicos / 12 hilos lógicos |
| **RAM** | 16 GB |
| **GPU** | NVIDIA GeForce GTX 1650 Mobile — 4 GiB GDDR6, 14 SMs, Compute Capability 7.5 |
| **Sistema operativo** | Windows 11 (build 10.0.26300) |
| **Entorno** | Python 3.11 en Conda (`nbody-gpu`) |
| **Librerías** (`requirements.txt`) | NumPy 2.4.6 · Numba 0.66.0 · Numba-CUDA 0.30.4 · Matplotlib 3.11.2 · psutil 7.2.2 |
| **CUDA Runtime** | 12.9 |

Versión de Numba con la que se hizo cada medición (la registra el propio script en `resultados/hardware*.txt`):

| Mediciones | Fecha | Numba | Fuente |
|:---|:---:|:---:|:---|
| CPU (`benchmark_cpu.py`, 4 corridas c1 a c4) | 2026-11-04 | 0.66.0 | `resultados/hardware.txt` |
| GPU (`benchmark_gpu.py`, 3 corridas) y verificación GPU | 2026-11-04 | 0.66.0 | `resultados/hardware_gpu.txt` |

Las mediciones de CPU y de GPU se hicieron el mismo día y con la misma versión de Numba (0.66.0), la fijada en `requirements.txt`. Entorno exacto: `conda list --export > entorno_conda.txt`.

---

## 4. Estructura del repositorio

```text
.
├── serial.py                 # Punto de entrada: simulación serial
├── paralelo.py               # Punto de entrada: simulación paralela (CPU / GPU)
├── medir.py                  # Script maestro: regenera CSV, tablas y figuras
├── requirements.txt          # Dependencias
├── README.md
├── src/                      # Módulos del modelo
│   ├── fisica.py             #   serial (NumPy)
│   ├── cpu_paralelo.py       #   Numba prange
│   ├── gpu_cuda.py           #   Numba CUDA
│   └── condiciones.py        #   condiciones iniciales
├── pruebas/                  # Validación física, benchmarks y renderizado
│   ├── test_entorno.py       #   comprueba Numba y CUDA
│   ├── prueba_serial.py      #   prueba rápida de la versión serial
│   ├── prueba_condiciones.py #   condiciones iniciales y estabilidad de una galaxia
│   ├── validacion_fisica.py  #   orden del integrador, reversibilidad y efecto de dt
│   ├── perfil_serial.py      #   perfilado del paso y fracción paralelizable medida
│   ├── benchmark_cpu.py      #   tiempos y verificación serial vs paralelo (CPU)
│   ├── benchmark_gpu.py      #   tiempos y verificación serial vs paralelo (GPU)
│   ├── analizar_cpu.py       #   speedup, eficiencia y Amdahl
│   ├── baselines_cpu.py      #   speedup frente a cada referencia
│   ├── resumen_repeticiones.py  # repeticiones por configuración
│   ├── resumen_gpu.py        #   consolida las corridas de GPU
│   └── visualizar.py         #   render de la colisión (imagen y video)
├── benchmarks/               # Datos crudos (CSV) de las mediciones
├── resultados/               # Gráficas, tablas consolidadas y diagnósticos
└── SIMULACIÓN WEB/           # Versión interactiva en el navegador (HTML)
```

---

## 5. Instrucciones de ejecución y reproducción

Activar previamente el entorno Conda:

```powershell
conda activate nbody-gpu
```

### A. Ejecución de simulaciones directas

```powershell
# Versión serial
python serial.py --n 500 --pasos 400

# Versión paralela en CPU (6 hilos)
python paralelo.py --backend cpu --hilos 6 --n 1000 --pasos 600

# Versión paralela en GPU
python paralelo.py --backend gpu --n 4000 --pasos 1000
```

`paralelo.py` hace un calentamiento antes de cronometrar (compilación JIT en CPU, arranque de CUDA en GPU) con un problema mínimo y reporta su duración aparte; el tiempo de la simulación no incluye la compilación. Ese tiempo sí incluye los registros periódicos de energía y momentos. Las mediciones formales de speedup están en `medir.py` (sección B), no en estos tiempos.

### B. Regenerar todo con `medir.py`

`medir.py` es el punto único para reproducir los CSV, las tablas y las figuras. Al terminar imprime un resumen con `[OK]` o `[FALLO]` por paso.

| Comando | Qué hace |
|:---|:---|
| `python medir.py --all` | Todo: CPU, GPU, tablas y figuras (incluye el video) |
| `python medir.py --all --rapido` | Lo mismo con tamaños reducidos; sirve para comprobar que todo corre. Las mediciones y el render van a `benchmarks/rapido/` y `resultados/rapido/` |
| `python medir.py --cpu` | Mediciones de CPU (4 corridas con pausa de 20 s para enfriar) y reanálisis de tablas |
| `python medir.py --gpu` | Mediciones de GPU (3 corridas con pausa; requiere CUDA) y reanálisis de tablas |
| `python medir.py --perfil` | Perfilado del paso de simulación (`resultados/tabla_perfil.csv`). También lo ejecutan `--cpu` y `--all` |
| `python medir.py --analizar` | Solo reanaliza los CSV existentes: speedup, eficiencia, Amdahl y tablas |
| `python medir.py --figuras` | Figuras de física y de la colisión, incluido el video |
| `python medir.py --figuras --sin-video` | Igual, sin renderizar el video (mucho más rápido) |

Opciones: `--corridas N` y `--pausa SEGUNDOS` (CPU, y pausa entre corridas de GPU) y `--corridas-gpu N` (por defecto 3).

Antes de medir (`--cpu`, `--gpu`, `--all`), `medir.py` compara las versiones instaladas con las de `requirements.txt` y muestra un aviso si no coinciden, para que ninguna medición quede hecha con versiones distintas de las documentadas. El aviso no detiene la ejecución.

### C. Qué comando genera cada archivo de resultados

Cada fila se puede ejecutar por separado (columna «Script») o, más cómodo, con `medir.py`.

| Archivo | Script | Con `medir.py` |
|:---|:---|:---:|
| `benchmarks/resultados_cpu_*_c*.csv`, `benchmarks/verificacion_cpu.csv`, `resultados/benchmark_cpu.png` (figura de la última corrida), `resultados/hardware.txt` | `python pruebas/benchmark_cpu.py` | `--cpu` |
| `benchmarks/resultados_gpu_corrida*.csv`, `benchmarks/resultados_gpu.csv` (última corrida), `benchmarks/verificacion_gpu.csv`, `resultados/benchmark_gpu.png`, `resultados/hardware_gpu.txt` | `python pruebas/benchmark_gpu.py` (solo la verificación: `--solo-verificacion`) | `--gpu` |
| `resultados/tabla_gpu.csv` (mediana y rango de las corridas de GPU) | `python pruebas/resumen_gpu.py` | `--analizar` |
| `resultados/tabla_perfil.csv` (fracción del paso en fuerzas, energía y resto; $f_{\text{medida}}$) | `python pruebas/perfil_serial.py` | `--perfil` (también `--cpu`, `--all`) |
| `resultados/analisis_cpu.png` (speedup y eficiencia con Amdahl), `resultados/tabla_speedup_cpu.csv` | `python pruebas/analizar_cpu.py` | `--analizar` |
| `resultados/tabla_baselines_cpu.csv` (speedup frente a cada referencia) | `python pruebas/baselines_cpu.py` | `--analizar` |
| `resultados/tabla_repeticiones.csv` | `python pruebas/resumen_repeticiones.py` | `--analizar` |
| `resultados/condiciones_iniciales.png`, `resultados/estabilidad_galaxia.png` | `python pruebas/prueba_condiciones.py` | `--figuras` |
| `resultados/validacion_dt.png` (orden 2, reversibilidad, efecto de $\Delta t$) | `python pruebas/validacion_fisica.py` (tarda unos minutos) | `--figuras` |
| `resultados/colision_final.png`, `resultados/colision.mp4` | `python pruebas/visualizar.py --backend gpu --n 4000 --pasos 6000` (con `--rapido` para una prueba corta) | `--figuras` |

> [!NOTE]
> - `visualizar.py` usa la GPU si hay CUDA (`--backend auto`); con `--backend cpu` o `numpy` corre sin GPU, más lento. Para guardar MP4 necesita `ffmpeg`; sin él, genera un GIF si está `imageio`.
> - Los scripts de `pruebas/` se pueden ejecutar desde cualquier carpeta.
> - Cada ejecución de `--cpu` escribe un lote nuevo de CSV (`resultados_cpu_<fecha>_c<k>.csv`). Al terminar todas las corridas, los CSV de sesiones anteriores se mueven (no se borran) a `benchmarks/archivo/<fecha>/`, de modo que `benchmarks/` contiene siempre un solo lote y el análisis no mezcla mediciones de sesiones distintas. Si una ejecución falla a medio camino, los datos anteriores siguen en su lugar.
> - Las pruebas con `--rapido` escriben las mediciones y el render en `benchmarks/rapido/` y `resultados/rapido/`, y el análisis las ignora. Las figuras de física (`prueba_condiciones.py` y `validacion_fisica.py`) no tienen modo rápido: regeneran las mismas figuras publicadas.
> - Comprobación del entorno y simulaciones sueltas: `python pruebas/test_entorno.py` y `python pruebas/prueba_serial.py`.

---

## 6. Resultados y síntesis de desempeño

### Definiciones

Con $T_1$ el tiempo con un hilo, $T_p$ el tiempo con $p$ hilos y $f$ la fracción paralelizable del tiempo serial:

```math
S_p = \frac{T_1}{T_p}, \qquad
E_p = \frac{S_p}{p}, \qquad
S_p^{\text{Amdahl}} = \frac{1}{(1 - f) + \dfrac{f}{p}}, \qquad
S_\infty = \frac{1}{1 - f}
```

En GPU no se habla de $p$ hilos: se reporta el speedup frente a la versión serial y frente a la CPU paralela (la eficiencia por hilo no aplica). El rendimiento del kernel se mide como:

```math
\text{Rendimiento} \;=\; \frac{N^{2}}{t_{\text{kernel}}} \quad \text{[interacciones/s]}
```

### Referencias del speedup (qué es $T_1$)

El speedup depende de la referencia con que se compare. En este proyecto hay tres, y no deben mezclarse:

| Ref. | Qué es | Dónde está |
|:---:|:---|:---|
| **A** | NumPy vectorizado, es decir el código de `serial.py` | `numpy` en `resultados_cpu*.csv` (medido solo hasta $N = 2000$) |
| **B** | Numba compilado a código máquina, 1 hilo, sin `prange` | `numba_1_hilo` (CPU) y `cpu_serial` (GPU) |
| **C** | El código paralelo ejecutado con $p$ hilos ($T_1$ es C con $p = 1$) | `numba_paralelo` y `numba_paralelo_hilos` |

- **Speedup de paralelismo** ($S_p = T_1/T_p$, criterio 5): usa $T_1$ = C con 1 hilo, medido en la misma ronda que $T_p$. Es la tabla `resultados/tabla_speedup_cpu.csv` y la figura `analisis_cpu.png`. B y C con 1 hilo son prácticamente el mismo código.
- **Frente a `serial.py`:** como `serial.py` es NumPy (A), la ganancia total frente a él mezcla dos efectos distintos: compilar a código máquina y paralelizar.

###  CPU multihilo (Numba)

#### Tiempos y speedup a 12 hilos

Mediana de 4 corridas (`python pruebas/baselines_cpu.py` los regenera en `resultados/tabla_baselines_cpu.csv`). $N$ es el número **total** de cuerpos; en `serial.py` y `paralelo.py`, `--n` es el número por galaxia.

| $N$ | A: NumPy (ms) | B: Numba 1 h (ms) | C: Numba 12 h (ms) | A/B (compilación) | B/C (paralelismo) | A/C (total) |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 2,80 | 0,407 | 0,102 | 10,9 | 3,7 | 30,2 |
| 500 | 12,13 | 0,604 | 0,172 | 20,1 | 3,6 | 73,7 |
| 1000 | 52,41 | 2,513 | 0,511 | 20,8 | 4,7 | 100,0 |
| 2000 | 197,11 | 9,982 | 2,407 | 19,7 | **4,3** | 83,2 |
| 4000 | no medido | 40,747 | 10,243 | — | 4,0 | — |
| 8000 | no medido | 167,267 | 39,203 | — | 4,3 | — |
| 16000 | no medido | 700,677 | 165,634 | — | **4,2** | — |

Frente a `serial.py` la versión paralela es unas **83 veces** más rápida ($N = 2000$), pero solo ~4,3× corresponde a paralelismo; el resto (~19,7×) viene de compilar con Numba. Por eso el análisis de eficiencia y de Amdahl usa B/C y no A/C.

#### Escalado por número de hilos

$T_1$ = mismo código con 1 hilo, calculado dentro de cada corrida; mediana de 4 corridas.

| Hilos $p$ | $S_p$ ($N = 2000$) | $E_p$ | $S_p$ ($N = 16000$) | $E_p$ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1,00 | 100 % | 1,00 | 100 % |
| 2 | 1,97 | 98 % | 1,94 | 97 % |
| 4 | 3,90 | 97 % | 3,43 | 86 % |
| **6** (núcleos físicos) | **4,96** | 83 % | **4,52** | 75 % |
| 8 ⚠️ | 3,77 | 47 % | 3,91 | 49 % |
| **12** (hilos lógicos) | **5,17** | 43 % | **4,31** | 36 % |

- **Speedup máximo con 12 hilos:** $5{,}17\times$ ($N = 2000$, rango 5,13 a 5,26) y $4{,}31\times$ ($N = 16000$), frente al mismo código paralelo con 1 hilo.
- **Con 6 hilos** (los núcleos físicos): $4{,}96\times$ ($N = 2000$) y $4{,}52\times$ ($N = 16000$).
- **Eficiencia:** 86 a 97 % con 4 hilos, 75 a 83 % con 6 hilos y 36 a 43 % con 12 hilos.

> [!WARNING]
> **La curva no es monótona.** Con 8 hilos el speedup cae a $3{,}77\times$ y $3{,}91\times$, por debajo del de 6 hilos, y vuelve a subir con 12. Ocurre en los dos tamaños y con rangos estrechos, así que no es ruido.
>
> **Hipótesis (no comprobada):** con 8 hilos sobre 6 núcleos físicos, dos núcleos ejecutan dos hilos cada uno; `prange` reparte las iteraciones en bloques iguales, los hilos que comparten núcleo van más lentos y los demás esperan al último.

#### Ajuste de Amdahl

Ajuste por mínimos cuadrados ($N \geq 2000$):

| Parámetro | Valor |
|:---|:---:|
| Fracción paralelizable efectiva $f$ | $\approx 0{,}88$ |
| Speedup máximo teórico $S_\infty = 1/(1-f)$ | $\approx 8{,}2$ |
| Predicción del modelo con 12 hilos | $\approx 5{,}1$ |
| $f_{\text{medida}}$ por perfilado | $\approx 0{,}998$ |

Es una $f$ *efectiva*: el ajuste es modesto porque la curva no es monótona (hundimiento con 8 hilos). El límite viene del hardware (6 núcleos físicos frente a 12 lógicos, frecuencia y contención de caché L3), no de código serial.

###  GPU CUDA (GTX 1650)

Mediana de 3 corridas; entre paréntesis, el rango mínimo – máximo. «vs B» y «vs C» son el cociente entre el tiempo de la CPU (referencia B: Numba 1 hilo; referencia C: Numba 12 hilos) y el del kernel; «n/d» indica que la CPU no se midió para ese $N$.

| $N$ | Kernel (ms) | Rango (ms) | vs B (×) | vs C (×) | Interacciones/s ($10^9$) | Transferencias PCIe (%) |
|---:|---:|:---:|:---:|:---:|---:|:---:|
| 1000 | **0,091** | 0,090 – 0,095 | **26,4** (25,6 – 26,9) | **5,7** (5,3 – 6,3) | 11,0 | **65,3** (65,2 – 66,1) |
| 2000 | **0,159** | 0,158 – 0,162 | **61,0** (59,9 – 61,2) | **12,5** (11,9 – 17,5) | 25,2 | **52,9** (52,3 – 53,1) |
| 4000 | **0,358** | 0,355 – 0,358 | **109,7** (109,0 – 112,2) | **22,7** (21,8 – 30,4) | 44,6 | **34,5** (34,5 – 34,8) |
| 8000 | **1,010** | 1,008 – 1,011 | **163,6** (159,8 – 171,0) | **38,8** (38,8 – 43,9) | 63,4 | **17,0** (16,7 – 17,0) |
| 16000 | **3,353** | 3,348 – 3,376 | **205,4** (193,7 – 212,8) | **50,2** (49,1 – 51,9) | 76,3 | **6,5** (6,5 – 6,8) |
| 32000 | **12,895** | 12,870 – 12,899 | n/d | n/d | 79,4 | **2,4** (2,4 – 2,8) |
| 64000 | **50,460** | 50,409 – 50,692 | n/d | n/d | 81,2 | **0,9** (0,9 – 1,0) |

- **Rendimiento:** el kernel alcanza ~$81 \times 10^{9}$ interacciones/s con $N = 64000$ (80,8 a 81,3 en las 3 corridas).
- **Con $N = 16000$**, el kernel es **205×** más rápido que Numba con 1 hilo (rango 194 a 213×) y **50×** más rápido que Numba con 12 hilos (rango 49 a 52×).
- **Variabilidad:** el tiempo del kernel varía menos del 2 % entre corridas; la variación de los cocientes viene sobre todo de la CPU (típico de un portátil con limitación térmica). Estas comparaciones usan la referencia B, no `serial.py` (NumPy).
- **Transferencias PCIe** (H2D + D2H) como fracción del tiempo total (kernel + transferencias): bajan del 5 % de forma consistente desde $N = 32000$. Los tiempos de kernel suponen que posiciones y velocidades permanecen en la GPU entre pasos, por lo que en la simulación completa las transferencias no se pagan en cada paso.

> [!IMPORTANT]
> **Precisión distinta (leer antes de interpretar el speedup de GPU).** El kernel trabaja en `float32` y todas las referencias de CPU en `float64`, así que estas razones de tiempo no comparan exactamente el mismo trabajo numérico. La diferencia favorece a la GPU: en tarjetas de consumo como la GTX 1650 (arquitectura Turing) la doble precisión es mucho más lenta que la simple (del orden de 32 veces menos rendimiento, según la especificación de la arquitectura), de modo que una GPU en `float64` daría un speedup bastante menor. En la CPU, `float32` también permitiría procesar el doble de datos por instrucción vectorial. Por eso el resultado debe leerse como «GPU en `float32` frente a CPU en `float64`»; la CPU en `float32` no se midió.

**Costo de usar `float32`:** el error relativo de las aceleraciones es de ~$10^{-6}$ a ~$7 \times 10^{-6}$ según $N$ (frente a ~$10^{-16}$ en CPU) y, como la dinámica es caótica, las trayectorias de GPU y CPU divergen con el tiempo aunque cada paso sea correcto. La conservación de energía en `float32` queda registrada en `benchmarks/verificacion_gpu.csv` y cumple el criterio del 1 %.

### Protocolo de medición (repeticiones y dispersión)

- Todos los tiempos se miden con `time.perf_counter` y la primera llamada (compilación JIT) se descarta.
- Una **corrida** es una ejecución completa e independiente de `benchmark_cpu.py` (o `benchmark_gpu.py`); se hicieron 4 de CPU y 3 de GPU, con pausas de enfriamiento. Las de CPU están en `benchmarks/` como `resultados_cpu_<fecha>_c1.csv` a `_c4.csv` y las de GPU como `resultados_gpu_corrida1.csv` a `_corrida3.csv`. Una **repetición** es una medición dentro de una corrida. La columna `n_corridas` de las tablas cuenta corridas, no repeticiones.
- Se reporta la mediana de cada corrida, luego la mediana entre corridas y el rango mínimo – máximo entre corridas (`analizar_cpu.py` y `resumen_gpu.py`).

| Bloque | Repeticiones por corrida | Corridas | Muestras por configuración |
|:---|:---:|:---:|:---:|
| CPU: escalado por hilos (speedup, eficiencia, Amdahl) | 15 | 4 | 60 |
| CPU: tiempo vs $N$, Numba 1 hilo y paralelo (refs. B y C) | 7 | 4 | 28 |
| CPU: NumPy, `serial.py` (ref. A) | 5 | 4 | 20 |
| GPU: kernel y transferencias | 15 | 3 | 45 |
| GPU: referencias de CPU (serial y paralelo) | 5 | 3 | 15 |

`python pruebas/resumen_repeticiones.py` regenera esta tabla desde los CSV (`resultados/tabla_repeticiones.csv`).

**Dispersión.** Entre las 4 corridas de CPU, el speedup con 12 hilos varía muy poco ($N = 16000$: 4,31 a 4,32; $N = 2000$: 5,13 a 5,26), pero el speedup B/C, que compara bloques medidos en momentos distintos, varía más ($N = 16000$: 3,78 a 4,32; $N = 2000$: 3,54 a 5,10). En GPU, el tiempo del kernel varía menos del 2 % entre las 3 corridas; la variación de los speedups de GPU viene sobre todo de los tiempos de CPU. Por eso los speedups se reportan con su rango y no como un único valor.

### Limitaciones

1. Todo se midió en un solo portátil, con límites térmicos y de frecuencia.
2. El speedup de GPU frente a la CPU solo se mide hasta $N = 16000$ (el serial tarda demasiado a $N$ mayor), y la GPU usa `float32` frente a `float64` en la CPU.
3. La caída del speedup con 8 hilos se explica con una hipótesis (reparto de iteraciones de `prange` y núcleos que comparten dos hilos) que no se comprobó con otras configuraciones de hilos.


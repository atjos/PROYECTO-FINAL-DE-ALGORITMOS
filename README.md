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
   - Error relativo máximo ~ 7 x 10^-7 en GPU (precisión simple float32).

---

## 3. Hardware y software usados

- CPU: Intel Core i5-11400H @ 2.70 GHz (6 núcleos físicos / 12 hilos lógicos)
- GPU: NVIDIA GeForce GTX 1650 Mobile (4 GiB GDDR6, 14 SMs, Compute Capability 7.5)
- Sistema Operativo: Windows 11 (build 10.0.26300)
- Entorno Python: Python 3.11 en Conda (nbody-gpu)
- Librerías principales: NumPy 2.4.6, Numba 0.66.0, Numba-CUDA 0.30.4, Matplotlib 3.11.2
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

### C. Validación física y visualización
- Validación de integrador (convergencia de orden 2 y dt):
  python pruebas/validacion_fisica.py
- Generar video/animación de la colisión (MP4/GIF):
  python pruebas/visualizar.py --backend gpu --n 4000 --pasos 6000

---

## 6. Resultados y síntesis de desempeño

1. CPU multihilo (Numba):
   - Speedup máximo obtenido de ~4.4x con 12 hilos frente a 1 hilo en la misma arquitectura.
   - Eficiencia paralela de ~73% a 4 hilos, descendiendo a ~37% a 12 hilos.
   - Ajuste por mínimos cuadrados de la ley de Amdahl arroja una fracción paralelizable efectiva f ~ 0.84. Esta cota está limitada por el factor de contención de memoria compartida L3 y la contención en Hyper-Threading (6 núcleos físicos vs 12 lógicos).
2. GPU CUDA (GTX 1650):
   - El kernel de GPU alcanza ~84 x 10^9 interacciones gravitatorias/segundo para N = 64000.
   - Con N = 16000, el kernel en GPU es ~220x más rápido que Numba 1 hilo y ~53x más rápido que Numba CPU con 12 hilos.
   - Para N >= 4000, el costo de transferencia de datos PCIe (H2D + D2H) cae por debajo del 5% del tiempo total, justificando la aceleración masiva en simulaciones continuas.
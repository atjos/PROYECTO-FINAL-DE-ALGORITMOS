from condiciones import crear_colision
import gpu_cuda as gpu
import cpu_paralelo as cpu
import fisica
import matplotlib.pyplot as plt
import matplotlib
from numba import cuda
import numba
import numpy as np
import sys
import os
import time
import csv
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))
matplotlib.use("Agg")

os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
os.makedirs(os.path.join(RAIZ, "benchmarks"), exist_ok=True)

RAPIDO = "--rapido" in sys.argv
EPS = 0.05
N_CORRECCION = 1000                  # N de la prueba de correccion de las fuerzas
# galaxias de la prueba de conservacion de energia (float32)
N_POR_GAL_ENERGIA = 1000
PASOS_ENERGIA = 6000                 # 6000 pasos de dt = 0.005  ->  t = 30
N_TPB = 16000
LISTA_TPB = [64, 128, 256, 512]
N_CPU_MAX = 16000                    # hasta aqui se compara con la CPU paralela
if RAPIDO:
    LISTA_N, REPS, PASOS_ENERGIA = [1000, 4000, 16000], 3, 1000
else:
    LISTA_N, REPS = [1000, 2000, 4000, 8000, 16000, 32000, 64000], 5


def datos(N, semilla=0):
    rng = np.random.default_rng(semilla)
    return rng.normal(size=(N, 3)), np.full(N, 1.0 / N)


def medir(f, reps=None):
    reps = REPS if reps is None else reps
    f()
    mejor = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


def medir_gpu(f, reps=None):
    reps = REPS if reps is None else reps
    f()
    # calentamiento (compila en la primera llamada)
    cuda.synchronize()
    mejor = np.inf
    for _ in range(reps):
        cuda.synchronize()
        t0 = time.perf_counter()
        f()
        cuda.synchronize()
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


def lanzador(N, tpb=gpu.TPB_POR_DEFECTO):
    """Prepara arrays en la GPU y devuelve una funcion que lanza SOLO el kernel de fuerzas."""
    pos, masas = datos(N)
    k_acc = gpu.construir_kernels(tpb)[0]
    d_pos4 = cuda.to_device(gpu.empaquetar(pos, masas))
    d_acc = cuda.device_array((N, 3), dtype=np.float32)
    eps2 = np.float32(EPS * EPS)
    bloques = (N + tpb - 1) // tpb
    return lambda: k_acc[bloques, tpb](d_pos4, d_acc, N, eps2)


def prueba_correccion():
    print(
        f"== 1) Correccion de las fuerzas: GPU (float32) vs NumPy (float64), N = {N_CORRECCION} ==")
    pos, masas = datos(N_CORRECCION)
    ref = fisica.aceleraciones(pos, masas, EPS)
    a = gpu.aceleraciones(pos, masas, EPS)
    print(
        f"  Error relativo maximo en a: {np.abs(a - ref).max() / np.abs(ref).max():.2e}  (float32: se espera ~1e-6 a 1e-5)")
    r = gpu.simular(pos, np.zeros_like(pos), masas,
                    EPS, 0.01, 0, guardar_pos=False)
    U_ref = fisica.energia(pos, np.zeros_like(pos), masas, EPS)[2]
    print(f"  Energia potencial: NumPy {U_ref:.6f}   GPU {r['E'][0]:.6f}")

    print("== 2) Dinamica: 400 pasos de la colision (N = 302), GPU vs referencia CPU float64 ==")
    p, v, m, _, _ = crear_colision(n_por_galaxia=150, semilla=1, eps=EPS)
    r_cpu = cpu.simular(p, v, m, EPS, 0.005, 400, cada=100, guardar_pos=False)
    r_gpu = gpu.simular(p, v, m, EPS, 0.005, 400, cada=100, guardar_pos=False)
    print(
        f"  Diferencia maxima de posicion tras 400 pasos: {np.abs(r_cpu['pos_final'] - r_gpu['pos_final']).max():.2e}")
    print(
        f"  Variacion de energia: CPU {abs(r_cpu['E'][-1] / r_cpu['E'][0] - 1):.2e}   GPU {abs(r_gpu['E'][-1] / r_gpu['E'][0] - 1):.2e}")


def prueba_energia_float32():
    n = 2 * N_POR_GAL_ENERGIA + 2
    print(
        f"== 3) Conservacion de energia en float32: colision con N = {n}, {PASOS_ENERGIA} pasos de dt = 0.005 ==")
    p, v, m, _, _ = crear_colision(
        n_por_galaxia=N_POR_GAL_ENERGIA, semilla=1, eps=EPS)
    t0 = time.perf_counter()
    r = gpu.simular(p, v, m, EPS, 0.005, PASOS_ENERGIA,
                    cada=100, guardar_pos=False)
    seg = time.perf_counter() - t0
    dE = np.abs(r["E"] - r["E"][0]) / abs(r["E"][0])
    L0, L1 = np.linalg.norm(r["L"][0]), np.linalg.norm(r["L"][-1])
    print(f"  max |dE/E| = {dE.max():.2e}   cambio relativo |L| = {abs(L1 - L0) / L0:.1e}   "
        f"|P| max = {np.linalg.norm(r['P'], axis=1).max():.1e}   ({seg:.1f} s en total)")
    print(
        f"  Criterio del 1 %: {'CUMPLE' if dE.max() < 0.01 else 'NO CUMPLE'}")


def prueba_rendimiento():
    dev = cuda.get_current_device().name
    dev = dev.decode() if isinstance(dev, bytes) else dev
    print(
        f"== 4) Tiempo de UN calculo de fuerzas vs N  (GPU: {dev}, bloque = {gpu.TPB_POR_DEFECTO}) ==")
    print(f"  {'N':>6} | {'CPU paralelo (s)':>16} | {'GPU (s)':>9} | {'speedup GPU/CPU':>15} | {'Ginteracciones/s':>16}")
    filas, t_g, t_c = [], {}, {}
    for N in LISTA_N:
        t_g[N] = medir_gpu(lanzador(N))
        filas.append(("gpu", N, gpu.TPB_POR_DEFECTO, t_g[N]))
        txt_c, txt_s = f"{'-':>16}", f"{'-':>15}"
        if N <= N_CPU_MAX:
            pos, masas = datos(N)
            t_c[N] = medir(lambda: cpu.aceleraciones_numba(
                pos, masas, EPS), reps=min(REPS, 3))
            filas.append(
                ("cpu_paralelo", N, numba.config.NUMBA_NUM_THREADS, t_c[N]))
            txt_c, txt_s = f"{t_c[N]:16.4f}", f"{t_c[N] / t_g[N]:15.1f}"
        print(
            f"  {N:6d} | {txt_c} | {t_g[N]:9.5f} | {txt_s} | {N * N / t_g[N] / 1e9:16.1f}")

    print(f"== 5) Efecto del tamano de bloque (N = {N_TPB}) ==")
    t_tpb = {}
    for tpb in LISTA_TPB:
        t_tpb[tpb] = medir_gpu(lanzador(N_TPB, tpb))
        filas.append(("gpu_tpb", N_TPB, tpb, t_tpb[tpb]))
        print(f"  hilos por bloque = {tpb:4d}: {t_tpb[tpb]:.5f} s")

    with open(os.path.join(RAIZ, "benchmarks", "resultados_gpu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementacion", "N", "hilos_o_tpb", "tiempo_s"])
        w.writerows(filas)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].loglog(list(t_c), list(t_c.values()),
                "o-", label="CPU paralela (Numba)")
    ax[0].loglog(LISTA_N, [t_g[N] for N in LISTA_N], "^-", label="GPU (CUDA)")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("tiempo por calculo de fuerzas (s)")
    ax[0].set_title("Tiempo vs N")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")
    ax[1].bar([str(t) for t in LISTA_TPB], [t_tpb[t] for t in LISTA_TPB])
    ax[1].set_xlabel("hilos por bloque")
    ax[1].set_ylabel("tiempo (s)")
    ax[1].set_title(f"Tamano de bloque (N = {N_TPB})")
    plt.tight_layout()
    plt.savefig(os.path.join(RAIZ, "resultados", "benchmark_gpu.png"), dpi=130)
    print("  Guardado: resultados/benchmark_gpu.png y benchmarks/resultados_gpu.csv")


if __name__ == "__main__":
    if not cuda.is_available():
        sys.exit(
            "CUDA no esta disponible en este entorno (revisa que este activo nbody-gpu).")
    prueba_correccion()
    prueba_energia_float32()
    prueba_rendimiento()

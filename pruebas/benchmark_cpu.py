from src.condiciones import crear_colision
from src import cpu_paralelo as cpu
from src import fisica
import matplotlib.pyplot as plt
import matplotlib
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
MAX_HILOS = numba.config.NUMBA_NUM_THREADS
if RAPIDO:
    LISTA_N, N_NUMPY_MAX, N_HILOS, REPS = [250, 500, 1000], 1000, 1000, 2
else:
    LISTA_N, N_NUMPY_MAX, N_HILOS, REPS = [
        250, 500, 1000, 2000, 4000, 8000, 16000], 2000, 8000, 5
LISTA_HILOS = [h for h in (1, 2, 4, 6, 8, 12, 16) if h <= MAX_HILOS]


def medir(f, *args, reps=REPS):
    """Mejor tiempo de 'reps' ejecuciones (la primera llamada, que compila, se hace aparte)."""
    f(*args)
    mejor = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        f(*args)
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


def datos(N, semilla=0):
    rng = np.random.default_rng(semilla)
    return rng.normal(size=(N, 3)), np.full(N, 1.0 / N)


def prueba_correccion():
    print("--- 1. Corrección numérica: paralelo vs referencia NumPy (N = 1000) ---")
    pos, masas = datos(1000)
    ref = fisica.aceleraciones(pos, masas, EPS)
    for nombre, f in (("numba 1 hilo", cpu.aceleraciones_numba_serial), ("numba paralelo", cpu.aceleraciones_numba)):
        a = f(pos, masas, EPS)
        err = np.abs(a - ref).max() / np.abs(ref).max()
        print(f"  {nombre:15s}: error relativo maximo en a = {err:.2e}")
    E_ref = fisica.energia(pos, np.zeros_like(pos), masas, EPS)[2]
    E_nb = cpu.energia(pos, np.zeros_like(pos), masas, EPS)[2]
    print(f"  Energia potencial: NumPy {E_ref:.8f}  Numba {E_nb:.8f}")

    print("--- 2. Corrección dinámica: 400 pasos de colisión (N = 302) ---")
    p, v, m, _, _ = crear_colision(n_por_galaxia=150, semilla=1, eps=EPS)
    r_ref = fisica.simular(p, v, m, EPS, 0.005, 400, cada=100)
    r_nb = cpu.simular(p, v, m, EPS, 0.005, 400, cada=100)
    print(
        f"  Diferencia maxima de posicion tras 400 pasos: {np.abs(r_ref['pos_final'] - r_nb['pos_final']).max():.2e}")
    print(
        f"  Variacion de energia: NumPy {abs(r_ref['E'][-1] / r_ref['E'][0] - 1):.2e}  Numba {abs(r_nb['E'][-1] / r_nb['E'][0] - 1):.2e}")


def prueba_rendimiento():
    print(
        f"--- 3. Tiempo de cálculo de fuerzas vs N (hilos disponibles: {MAX_HILOS}) ---")
    numba.set_num_threads(MAX_HILOS)
    filas, t_np, t_s, t_p = [], {}, {}, {}
    print(f"  {'N':>6} | {'NumPy (s)':>10} | {'Numba 1h (s)':>12} | {'Numba paralelo (s)':>18} | {'Mints/s paralelo':>16}")
    for N in LISTA_N:
        pos, masas = datos(N)
        t_s[N] = medir(cpu.aceleraciones_numba_serial, pos, masas, EPS)
        t_p[N] = medir(cpu.aceleraciones_numba, pos, masas, EPS)
        if N <= N_NUMPY_MAX:
            t_np[N] = medir(fisica.aceleraciones, pos,
                            masas, EPS, reps=min(REPS, 3))
            filas.append(("numpy", N, 1, t_np[N]))
        filas.append(("numba_1_hilo", N, 1, t_s[N]))
        filas.append(("numba_paralelo", N, MAX_HILOS, t_p[N]))
        txt_np = f"{t_np[N]:10.4f}" if N in t_np else f"{'-':>10}"
        print(
            f"  {N:6d} | {txt_np} | {t_s[N]:12.4f} | {t_p[N]:18.4f} | {N * N / t_p[N] / 1e6:16.0f}")

    print(f"--- 4. Escalabilidad y aceleración vs número de hilos (N = {N_HILOS}) ---")
    pos, masas = datos(N_HILOS)
    tiempos_h = {}
    for h in LISTA_HILOS:
        numba.set_num_threads(h)
        tiempos_h[h] = medir(cpu.aceleraciones_numba, pos, masas, EPS)
        filas.append(("numba_paralelo_hilos", N_HILOS, h, tiempos_h[h]))
    base = medir(cpu.aceleraciones_numba_serial, pos, masas, EPS)
    numba.set_num_threads(MAX_HILOS)
    print(f"  Referencia (Numba compilado, 1 hilo): {base:.4f} s")
    print(
        f"  {'hilos':>5} | {'tiempo (s)':>10} | {'speedup':>8} | {'eficiencia':>10}")
    for h in LISTA_HILOS:
        sp = base / tiempos_h[h]
        print(f"  {h:5d} | {tiempos_h[h]:10.4f} | {sp:8.2f} | {sp / h:10.0%}")
    if t_np:
        n_ref = max(t_np)
        print(
            f"  Speedup paralelo vs NumPy (N = {n_ref}): {t_np[n_ref] / t_p[n_ref]:.1f}x")

    with open(os.path.join(RAIZ, "benchmarks", "resultados_cpu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementacion", "N", "hilos", "tiempo_s"])
        w.writerows(filas)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].loglog(list(t_np), list(t_np.values()), "o-", label="NumPy (serial)")
    ax[0].loglog(LISTA_N, [t_s[N]
                for N in LISTA_N], "s-", label="Numba, 1 hilo")
    ax[0].loglog(LISTA_N, [t_p[N] for N in LISTA_N],
                "^-", label=f"Numba, {MAX_HILOS} hilos")
    ax[0].set_xlabel("Número de partículas (N)")
    ax[0].set_ylabel("Tiempo por cálculo de fuerzas (s)")
    ax[0].set_title("Tiempo de cálculo vs. N")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")
    ax[1].plot(LISTA_HILOS, [base / tiempos_h[h]
            for h in LISTA_HILOS], "o-", label="medido")
    ax[1].plot(LISTA_HILOS, LISTA_HILOS, "k--", lw=0.8, label="ideal")
    ax[1].set_xlabel("Número de hilos")
    ax[1].set_ylabel("Aceleración (speedup)")
    ax[1].set_title(f"Escalabilidad multihilo (N = {N_HILOS})")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RAIZ, "resultados", "benchmark_cpu.png"), dpi=130)
    print("  Guardado: resultados/benchmark_cpu.png y benchmarks/resultados_cpu.csv")


if __name__ == "__main__":
    prueba_correccion()
    prueba_rendimiento()

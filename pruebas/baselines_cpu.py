"""
Speedup de la CPU frente a CADA referencia posible (T1 distinto en cada caso).

Uso:
    python pruebas/baselines_cpu.py        # lee benchmarks/resultados_cpu*.csv

Referencias (todas miden el calculo de aceleraciones, N = particulas totales):
    A. NumPy vectorizado        -> es el codigo de serial.py (fisica.aceleraciones)
    B. Numba compilado, 1 hilo  -> mismo bucle que el paralelo, sin prange
    C. Numba paralelo, p hilos  -> version paralela (p = maximo de hilos)

Speedups que reporta:
    S_compilacion  = A / B   (NO es paralelismo: es la ganancia de compilar a codigo maquina)
    S_paralelismo  = B / C   (paralelismo puro: lo que mide el criterio 5 de la rubrica)
    S_total        = A / C   (lo que gana el usuario frente a serial.py; mezcla las dos anteriores)
Como en analizar_cpu.py: mediana por corrida y luego mediana (y min-max) entre corridas.
NumPy solo se mide hasta N_NUMPY_MAX, por eso A solo existe para N pequenos.
Salida: resultados/tabla_baselines_cpu.csv
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def leer(ruta):
    d = defaultdict(list)
    with open(ruta, newline="") as f:
        for r in csv.DictReader(f):
            d[(r["implementacion"], int(r["N"]), int(r["hilos"]))].append(float(r["tiempo_s"]))
    return {k: float(np.median(v)) for k, v in d.items()}


def resumen(valores):
    return float(np.median(valores)), min(valores), max(valores)


def main():
    rutas = sorted(glob.glob(os.path.join(RAIZ, "benchmarks", "resultados_cpu*.csv")))
    corridas = [leer(r) for r in rutas]
    hmax = max(h for c in corridas for (i, n, h) in c if i == "numba_paralelo")
    Ns = sorted({n for c in corridas for (i, n, h) in c if i == "numba_paralelo"})

    filas = []
    print(f"Corridas: {len(corridas)} | hilos del paralelo: {hmax}")
    print(f"{'N':>6} | {'A NumPy (ms)':>12} | {'B 1h (ms)':>10} | {'C ' + str(hmax) + 'h (ms)':>9} | "
          f"{'A/B compil.':>11} | {'B/C paralel.':>12} | {'A/C total':>9}")
    for N in Ns:
        A = [c[("numpy", N, 1)] for c in corridas if ("numpy", N, 1) in c]
        B = [c[("numba_1_hilo", N, 1)] for c in corridas if ("numba_1_hilo", N, 1) in c]
        C = [c[("numba_paralelo", N, hmax)] for c in corridas if ("numba_paralelo", N, hmax) in c]
        sp_par = resumen([b / p for b, p in zip(B, C)])
        if A and len(A) == len(B) == len(C):
            sp_comp = resumen([a / b for a, b in zip(A, B)])
            sp_tot = resumen([a / p for a, p in zip(A, C)])
            a_ms = np.median(A) * 1e3
        else:
            sp_comp = sp_tot = (float("nan"),) * 3
            a_ms = float("nan")
        filas.append((N, len(C), a_ms, np.median(B) * 1e3, np.median(C) * 1e3,
                      sp_comp[0], sp_par[0], sp_par[1], sp_par[2], sp_tot[0]))
        print(f"{N:6d} | {a_ms:12.3f} | {np.median(B) * 1e3:10.3f} | {np.median(C) * 1e3:9.3f} | "
              f"{sp_comp[0]:11.1f} | {sp_par[0]:12.2f} | {sp_tot[0]:9.1f}")

    with open(os.path.join(RAIZ, "resultados", "tabla_baselines_cpu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "n_corridas", "t_numpy_ms", "t_numba_1h_ms", f"t_numba_{hmax}h_ms",
                    "S_compilacion_A_B", "S_paralelismo_B_C", "S_paralelismo_min", "S_paralelismo_max",
                    "S_total_A_C"])
        w.writerows(filas)
    print("\nGuardado: resultados/tabla_baselines_cpu.csv")


if __name__ == "__main__":
    main()

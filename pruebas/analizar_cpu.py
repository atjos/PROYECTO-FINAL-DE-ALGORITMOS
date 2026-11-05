"""
Analisis de las corridas de benchmark_cpu.py (puede haber varias).

Uso:
    python benchmarks/analizar_cpu.py                       # lee benchmarks/resultados_cpu*.csv
    python benchmarks/analizar_cpu.py a.csv b.csv c.csv     # o archivos concretos

Cada CSV es UNA corrida independiente. Para cada corrida se calcula la mediana de sus repeticiones;
luego se resume ENTRE corridas (mediana y rango minimo-maximo). Asi la dispersion que se reporta
incluye la variabilidad real del equipo entre una ejecucion y otra, no solo el ruido dentro de una.

Salidas:
    resultados/tabla_speedup_cpu.csv   speedup, eficiencia y f por (N, hilos)
    resultados/analisis_cpu.png        speedup y eficiencia con curva de Amdahl ajustada
Requiere: numpy, matplotlib (no requiere numba).
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
# por debajo de este N el tiempo es < 1 ms y domina el ruido / costo de hilos
N_MIN_AJUSTE = 2000


def leer(ruta):
    d = defaultdict(list)
    with open(ruta, newline="") as f:
        for r in csv.DictReader(f):
            d[(r["implementacion"], int(r["N"]), int(r["hilos"]))].append(
                float(r["tiempo_s"]))
    return {k: float(np.median(v)) for k, v in d.items()}


def amdahl(p, f):
    return 1.0 / ((1.0 - f) + f / p)


def ajustar_f(hilos, speedups):
    """f que minimiza el error cuadratico entre S medido y Amdahl (busqueda en rejilla)."""
    fs = np.linspace(0.0, 0.999, 2000)
    err = [np.sum((np.array(speedups) - amdahl(np.array(hilos), f)) ** 2)
           for f in fs]
    return float(fs[int(np.argmin(err))])


def main():
    rutas = sys.argv[1:] or sorted(
        glob.glob(os.path.join(RAIZ, "benchmarks", "resultados_cpu*.csv")))
    if not rutas:
        sys.exit("No encontre CSV. Corre primero benchmark_cpu.py")
    corridas = [leer(r) for r in rutas]
    print(f"Corridas analizadas: {len(corridas)}")
    for r in rutas:
        print("  -", os.path.basename(r))

    # A. Speedup 1 hilo -> max hilos vs N
    Ns = sorted({n for c in corridas for (impl, n, h)
                in c if impl == "numba_paralelo"})
    hmax = max(h for c in corridas for (impl, n, h)
               in c if impl == "numba_paralelo")
    print(
        f"\nA. Speedup Numba 1 hilo -> {hmax} hilos vs N (mediana entre corridas; rango min-max)")
    print(f"  {'N':>6} | {'speedup':>8} | {'min-max':>13}")
    for N in Ns:
        s = [c[("numba_1_hilo", N, 1)] / c[("numba_paralelo", N, hmax)]
             for c in corridas if ("numba_paralelo", N, hmax) in c]
        print(f"  {N:6d} | {np.median(s):8.2f} | {min(s):5.2f} - {max(s):5.2f}")

    # B. Escalado por hilos
    Ns_esc = sorted({n for c in corridas for (impl, n, h)
                    in c if impl == "numba_paralelo_hilos"})
    hilos = sorted({h for c in corridas for (impl, n, h)
                   in c if impl == "numba_paralelo_hilos"})
    filas, resumen = [], {}
    print("\nB. Escalado por hilos: S = T1/Tp con T1 = mismo codigo con 1 hilo, calculado dentro de cada corrida")
    for N in Ns_esc:
        print(f"  N = {N}")
        print(
            f"  {'hilos':>5} | {'speedup':>8} | {'min-max':>13} | {'efic.':>6} | {'f Amdahl':>8}")
        resumen[N] = []
        for h in hilos:
            s = [c[("numba_paralelo_hilos", N, 1)] / c[("numba_paralelo_hilos", N, h)]
                 for c in corridas if ("numba_paralelo_hilos", N, h) in c]
            sm, smin, smax = float(np.median(s)), min(s), max(s)
            ef = sm / h
            fa = (1 - 1 / sm) / (1 - 1 / h) if h > 1 else float("nan")
            resumen[N].append((h, sm, smin, smax, ef, fa))
            filas.append((N, h, len(s), sm, smin, smax, ef, fa))
            txt = f"{fa:8.3f}" if h > 1 else f"{'-':>8}"
            print(
                f"  {h:5d} | {sm:8.2f} | {smin:5.2f} - {smax:5.2f} | {ef:6.0%} | {txt}")

    # C. Ajuste de Amdahl con N grandes
    grandes = [N for N in Ns_esc if N >= N_MIN_AJUSTE]
    h_aj, s_aj = [], []
    for N in grandes:
        for (h, sm, *_r) in resumen[N]:
            h_aj.append(h)
            s_aj.append(sm)
    f_aj = ajustar_f(h_aj, s_aj)
    print(
        f"\nC. Ajuste de Amdahl (N >= {N_MIN_AJUSTE}): f = {f_aj:.3f}  ->  speedup maximo teorico 1/(1-f) = {1 / (1 - f_aj):.1f}")
    print(
        f"   Prediccion del modelo con {hmax} hilos: {amdahl(hmax, f_aj):.2f}")
    print("   OJO: f es una fraccion EFECTIVA. El calculo de fuerzas esta 100 % dentro de prange; lo que limita")
    print("   es el hardware (nucleos fisicos vs logicos, frecuencia, memoria), no codigo serial.")

    with open(os.path.join(RAIZ, "resultados", "tabla_speedup_cpu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "hilos", "n_corridas", "speedup_mediana",
                   "speedup_min", "speedup_max", "eficiencia", "f_amdahl"])
        w.writerows(filas)

    # Gráfica de speedup y eficiencia
    os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    p = np.linspace(1, max(hilos), 100)
    ax[0].plot(p, p, "k--", lw=0.8, label="ideal")
    ax[0].plot(p, amdahl(p, f_aj), "r:", lw=1.5,
               label=f"Amdahl ajustado (f = {f_aj:.2f})")
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8, label="ideal")
    ax[1].plot(p, amdahl(p, f_aj) / p, "r:", lw=1.5,
               label=f"Amdahl ajustado (f = {f_aj:.2f})")
    for N in Ns_esc:
        h = [d[0] for d in resumen[N]]
        sm = np.array([d[1] for d in resumen[N]])
        lo = sm - np.array([d[2] for d in resumen[N]])
        hi = np.array([d[3] for d in resumen[N]]) - sm
        ax[0].errorbar(h, sm, yerr=[lo, hi], fmt="o-",
                       capsize=3, label=f"medido, N = {N}")
        ax[1].errorbar(h, sm / np.array(h), yerr=[lo / np.array(h), hi / np.array(h)], fmt="o-", capsize=3,
                       label=f"medido, N = {N}")
    ax[0].set_ylabel("Speedup  S = T1 / Tp")
    ax[0].set_title(
        f"Escalabilidad multihilo ({len(corridas)} corrida(s); barras = min-max)")
    ax[1].set_ylabel("Eficiencia  E = S / p")
    ax[1].set_ylim(0, 1.1)
    ax[1].set_title("Eficiencia paralela")
    for a in ax:
        a.set_xlabel("Numero de hilos (p)")
        a.grid(alpha=0.3)
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RAIZ, "resultados", "analisis_cpu.png"), dpi=130)
    print("\nGuardado: resultados/analisis_cpu.png y resultados/tabla_speedup_cpu.csv")


if __name__ == "__main__":
    main()

"""
resumen_gpu.py - Consolida varias corridas de GPU: mediana entre corridas y rango min-max.

Lee benchmarks/resultados_gpu_corrida*.csv (si no hay, usa benchmarks/resultados_gpu.csv) y escribe
resultados/tabla_gpu.csv con, por tamano N:
  - tiempo del kernel (mediana entre corridas y rango),
  - speedup frente a Numba 1 hilo (ref. B) y frente a Numba 12 hilos (ref. C),
  - porcentaje del tiempo total que se va en transferencias PCIe (H2D + D2H).

Uso:  python pruebas/resumen_gpu.py
"""
import csv
import glob
import os
from collections import defaultdict
from statistics import median

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_BENCH = os.path.join(RAIZ, "benchmarks")
DIR_RES = os.path.join(RAIZ, "resultados")


def leer(ruta):
    """Devuelve {(implementacion, N): mediana de las repeticiones} de una corrida."""
    datos = defaultdict(list)
    with open(ruta, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            datos[(fila["implementacion"], int(fila["N"]))].append(float(fila["tiempo_s"]))
    return {k: median(v) for k, v in datos.items()}


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_BENCH, "resultados_gpu_corrida*.csv")))
    if not rutas:
        rutas = [os.path.join(DIR_BENCH, "resultados_gpu.csv")]
    corridas = [leer(r) for r in rutas]
    print(f"Corridas de GPU usadas: {len(corridas)}")
    for r in rutas:
        print("  -", os.path.basename(r))

    tamanos = sorted({n for c in corridas for (impl, n) in c if impl == "gpu_kernel"})

    def por_corrida(impl, n, f=lambda x: x):
        return [f(c[(impl, n)]) for c in corridas if (impl, n) in c]

    filas = []
    for n in tamanos:
        k = por_corrida("gpu_kernel", n)
        # speedups y % PCIe calculados DENTRO de cada corrida (mismas condiciones termicas)
        s_b, s_c, pcie = [], [], []
        for c in corridas:
            if ("gpu_kernel", n) not in c:
                continue
            ker = c[("gpu_kernel", n)]
            if ("cpu_serial", n) in c:
                s_b.append(c[("cpu_serial", n)] / ker)
            if ("cpu_paralelo", n) in c:
                s_c.append(c[("cpu_paralelo", n)] / ker)
            if ("gpu_h2d", n) in c and ("gpu_d2h", n) in c:
                tr = c[("gpu_h2d", n)] + c[("gpu_d2h", n)]
                pcie.append(100 * tr / (ker + tr))

        def res(v):
            return (median(v), min(v), max(v)) if v else (float("nan"),) * 3

        filas.append([n, len(k), *res([x * 1e3 for x in k]), *res(s_b), *res(s_c), *res(pcie)])

    cab = ["N", "n_corridas",
           "kernel_ms_mediana", "kernel_ms_min", "kernel_ms_max",
           "speedup_vs_B_mediana", "speedup_vs_B_min", "speedup_vs_B_max",
           "speedup_vs_C_mediana", "speedup_vs_C_min", "speedup_vs_C_max",
           "pcie_pct_mediana", "pcie_pct_min", "pcie_pct_max"]
    os.makedirs(DIR_RES, exist_ok=True)
    salida = os.path.join(DIR_RES, "tabla_gpu.csv")
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cab)
        w.writerows([[round(x, 4) if isinstance(x, float) else x for x in fila] for fila in filas])

    print(f"\n{'N':>6} | {'kernel ms (min-max)':>24} | {'S vs B (min-max)':>22} | {'S vs C (min-max)':>20} | {'% PCIe (min-max)':>18}")
    for fila in filas:
        n, nc, km, kmin, kmax, bm, bmin, bmax, cm, cmin, cmax, pm, pmin, pmax = fila
        print(f"{n:6d} | {km:8.3f} ({kmin:.3f}-{kmax:.3f})   | {bm:7.1f} ({bmin:.0f}-{bmax:.0f})     "
              f"| {cm:6.1f} ({cmin:.0f}-{cmax:.0f})   | {pm:5.1f} ({pmin:.1f}-{pmax:.1f})")
    print(f"\nGuardado: {salida}")


if __name__ == "__main__":
    main()

"""
Cuenta cuantas repeticiones respaldan cada medicion (para verificar el criterio ">= 5 repeticiones").

Uso:
    python pruebas/resumen_repeticiones.py
Lee benchmarks/resultados_cpu*.csv y benchmarks/resultados_gpu_corrida*.csv (o resultados_gpu.csv).
Salida: resultados/tabla_repeticiones.csv

Definiciones: una "corrida" es una ejecucion completa e independiente de benchmark_cpu.py (un CSV);
una "repeticion" es una medicion dentro de la corrida. La mediana de cada corrida usa solo las
repeticiones de esa corrida; las corridas se combinan despues (ver analizar_cpu.py).
"""
import csv
import glob
import os
from collections import Counter, defaultdict

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MINIMO = 5

BLOQUES_CPU = {
    "numba_paralelo_hilos": "CPU: escalado por hilos (speedup, eficiencia, Amdahl)",
    "numba_1_hilo": "CPU: tiempo vs N, Numba 1 hilo (ref. B)",
    "numba_paralelo": "CPU: tiempo vs N, Numba paralelo (ref. C)",
    "numpy": "CPU: NumPy, serial.py (ref. A)",
}
BLOQUES_GPU = {
    "gpu_kernel": "GPU: kernel", "gpu_h2d": "GPU: copia host->device", "gpu_d2h": "GPU: copia device->host",
    "cpu_serial": "GPU: referencia CPU serial", "cpu_paralelo": "GPU: referencia CPU paralelo",
}


def contar(rutas, bloques):
    reps = defaultdict(Counter)          # implementacion -> {reps por (archivo, config): ...}
    for ruta in rutas:
        por_cfg = defaultdict(int)
        with open(ruta, newline="") as f:
            for r in csv.DictReader(f):
                por_cfg[(r["implementacion"], r["N"], r.get("hilos") or r.get("hilos_o_tpb"))] += 1
        for (impl, _n, _h), k in por_cfg.items():
            if impl in bloques:
                reps[impl][k] += 1
    return reps


def main():
    cpu = sorted(glob.glob(os.path.join(RAIZ, "benchmarks", "resultados_cpu*.csv")))
    # varias corridas de GPU: resultados_gpu_corrida<k>.csv (si no hay, el resultados_gpu.csv unico)
    gpu = sorted(glob.glob(os.path.join(RAIZ, "benchmarks", "resultados_gpu_corrida*.csv")))
    if not gpu:
        gpu = [os.path.join(RAIZ, "benchmarks", "resultados_gpu.csv")]
    filas = []
    for rutas, bloques in ((cpu, BLOQUES_CPU), (gpu, BLOQUES_GPU)):
        reps = contar(rutas, bloques)
        for impl, nombre in bloques.items():
            if impl not in reps:
                continue
            k = min(reps[impl])                       # peor caso del bloque
            filas.append((nombre, impl, k, len(rutas), k * len(rutas), "si" if k >= MINIMO else "NO"))
    print(f"{'bloque':58} | {'rep/corrida':>11} | {'corridas':>8} | {'muestras':>8} | >=5 por corrida")
    for f in filas:
        print(f"{f[0]:58} | {f[2]:11d} | {f[3]:8d} | {f[4]:8d} | {f[5]}")
    with open(os.path.join(RAIZ, "resultados", "tabla_repeticiones.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bloque", "implementacion", "repeticiones_por_corrida", "corridas",
                    "muestras_por_configuracion", "cumple_5_por_corrida"])
        w.writerows(filas)
    print("\nGuardado: resultados/tabla_repeticiones.csv")


if __name__ == "__main__":
    main()

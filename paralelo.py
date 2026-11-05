"""
paralelo.py - Implementación y punto de entrada de la simulación paralela (CPU y GPU).

Proyecto final de Cálculo Científico - Universidad Industrial de Santander.
Cumple con el entregable 2.1 de la guía del proyecto.

Uso:
    python paralelo.py                     # Ejecuta por defecto en CPU con todos los núcleos
    python paralelo.py --backend cpu --hilos 6   # Ejecuta en CPU fijando 6 hilos
    python paralelo.py --backend gpu       # Ejecuta en GPU CUDA (si está disponible)
    python paralelo.py --n 1000 --pasos 1000  # Configura tamaño de problema y pasos

Antes de cronometrar se hace un calentamiento (compilacion JIT y arranque de la GPU) con un problema
minimo; su duracion se reporta aparte y NO entra en el tiempo de la simulacion.
"""

import sys
import os
import time
import argparse
import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)

from src.condiciones import crear_colision

from src import cpu_paralelo as cpu
try:
    from src import gpu_cuda as gpu
    from numba import cuda
    GPU_DISPONIBLE = cuda.is_available()
except Exception:
    GPU_DISPONIBLE = False

def main():
    parser = argparse.ArgumentParser(description="Simulación gravitacional de galaxias en versiones paralelas (CPU/GPU).")
    parser.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="auto",
                        help="Backend de paralelización: cpu, gpu o auto (default: auto)")
    parser.add_argument("--hilos", type=int, default=None,
                        help="Número de hilos de CPU (solo aplica si backend=cpu)")
    parser.add_argument("--n", type=int, default=1000, help="Partículas por galaxia (default: 1000)")
    parser.add_argument("--pasos", type=int, default=600, help="Pasos de integración (default: 600)")
    parser.add_argument("--dt", type=float, default=0.005, help="Paso de tiempo dt (default: 0.005)")
    parser.add_argument("--eps", type=float, default=0.05, help="Parámetro de suavizado (default: 0.05)")
    args = parser.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "gpu" if GPU_DISPONIBLE else "cpu"

    if backend == "gpu" and not GPU_DISPONIBLE:
        print("[AVISO] GPU CUDA no disponible. Cambiando automáticamente a backend CPU.", file=sys.stderr)
        backend = "cpu"

    print("=" * 65)
    print(f"SIMULACIÓN PARALELA - Backend: {backend.upper()}")
    if backend == "cpu":
        import numba
        if args.hilos:
            numba.set_num_threads(args.hilos)
        hilos_usados = numba.get_num_threads()
        print(f"Hardware: CPU Numba prange | Hilos activos: {hilos_usados}")
    else:
        dev = cuda.get_current_device()
        dev_nombre = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
        print(f"Hardware: GPU CUDA | Dispositivo: {dev_nombre} | Precisión: float32")
    print(f"Configuración: N_por_galaxia={args.n} | Pasos={args.pasos} | dt={args.dt} | eps={args.eps}")
    print("=" * 65)

    print("\n[1/4] Generando condiciones iniciales de colisión...")
    pos, vel, masas, gal, nuc = crear_colision(n_por_galaxia=args.n, eps=args.eps, semilla=42)
    n_total = len(masas)
    print(f"      Partículas totales: {n_total}")

    # Calentamiento fuera del cronómetro: la compilación JIT y el arranque CUDA
    # dependen de los tipos de argumentos, no del tamaño (64 cuerpos y 2 pasos bastan).
    print("\n[2/4] Calentamiento (compilación JIT / arranque de la GPU; no cuenta en el tiempo)...")
    simular = gpu.simular if backend == "gpu" else cpu.simular
    k = min(n_total, 64)
    t0 = time.perf_counter()
    simular(pos[:k], vel[:k], masas[:k], args.eps, args.dt, 2, cada=1, guardar_pos=False)
    if backend == "gpu":
        cuda.synchronize()
    t_calentamiento = time.perf_counter() - t0
    print(f"      Calentamiento: {t_calentamiento:.2f} s")

    print(f"\n[3/4] Integrando {args.pasos} pasos temporales...")
    t0 = time.perf_counter()
    if backend == "gpu":
        res = gpu.simular(pos, vel, masas, args.eps, args.dt, args.pasos, cada=max(1, args.pasos // 10), guardar_pos=False)
    else:
        res = cpu.simular(pos, vel, masas, args.eps, args.dt, args.pasos, cada=max(1, args.pasos // 10), guardar_pos=False)
    tiempo_total = time.perf_counter() - t0   # simular() termina con cuda.synchronize() en GPU

    dE = np.abs(res["E"] - res["E"][0]) / np.abs(res["E"][0])
    p_max = np.linalg.norm(res["P"], axis=1).max()
    L0 = np.linalg.norm(res["L"][0])
    L_fin = np.linalg.norm(res["L"][-1])
    dL = np.abs(L_fin - L0) / L0

    print("\n[4/4] Métricas de conservación y rendimiento:")
    print(f"      - Calentamiento (JIT/GPU):  {t_calentamiento:.3f} s  (no incluido en los tiempos siguientes)")
    print(f"      - Tiempo total de cómputo:  {tiempo_total:.3f} s  (incluye los registros de energía y diagnósticos)")
    print(f"      - Tiempo por paso:          {tiempo_total / args.pasos * 1e3:.2f} ms")
    print(f"      - Rendimiento interacciones:{n_total * n_total / (tiempo_total / args.pasos) / 1e6:.1f} M-Interacciones/s")
    print(f"      - Error relativo máx. |ΔE|: {dE.max():.2e}  ({'CUMPLE < 1%' if dE.max() < 0.01 else 'REVISAR'})")
    print(f"      - Variación relativa de |L|: {dL:.2e}")
    print(f"      - Momento lineal máx. |P|:  {p_max:.2e}")
    print("=" * 65)

if __name__ == "__main__":
    main()
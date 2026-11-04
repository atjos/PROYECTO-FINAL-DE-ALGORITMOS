"""
serial.py - Implementación y punto de entrada de la simulación serial (NumPy).

Proyecto final de Cálculo Científico - Universidad Industrial de Santander.
Cumple con el entregable 2.1 de la guía del proyecto.

Uso:
    python serial.py                 # Ejecuta una simulación serial de prueba y reporta métricas
    python serial.py --n 500         # Permite configurar el número de partículas por galaxia
    python serial.py --pasos 500     # Número de pasos de integración
"""

import sys
import os
import time
import argparse
import numpy as np

# Asegurar importación de src/
RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)

from src.fisica import aceleraciones, energia, paso_leapfrog, simular
from src.condiciones import crear_colision

def main():
    parser = argparse.ArgumentParser(description="Simulación gravitacional de galaxias en versión serial (NumPy).")
    parser.add_argument("--n", type=int, default=300, help="Partículas por galaxia (default: 300)")
    parser.add_argument("--pasos", type=int, default=400, help="Pasos de integración (default: 400)")
    parser.add_argument("--dt", type=float, default=0.005, help="Paso de tiempo dt (default: 0.005)")
    parser.add_argument("--eps", type=float, default=0.05, help="Parámetro de suavizado (default: 0.05)")
    args = parser.parse_args()

    print("=" * 65)
    print("SIMULACIÓN SERIAL (NumPy puro)")
    print(f"Configuración: N_por_galaxia={args.n} | Pasos={args.pasos} | dt={args.dt} | eps={args.eps}")
    print("=" * 65)

    # 1. Crear condiciones iniciales
    print("\n[1/3] Generando condiciones iniciales de colisión...")
    pos, vel, masas, gal, nuc = crear_colision(n_por_galaxia=args.n, eps=args.eps, semilla=42)
    n_total = len(masas)
    print(f"      Partículas totales: {n_total} (incluye núcleos masivos)")

    # 2. Ejecutar simulación
    print(f"\n[2/3] Integrando {args.pasos} pasos temporales (Leapfrog Kick-Drift-Kick)...")
    t0 = time.perf_counter()
    res = simular(pos, vel, masas, args.eps, args.dt, args.pasos, cada=max(1, args.pasos // 10))
    tiempo_total = time.perf_counter() - t0

    # 3. Reporte de física y tiempos
    dE = np.abs(res["E"] - res["E"][0]) / np.abs(res["E"][0])
    p_max = np.linalg.norm(res["P"], axis=1).max()
    L0 = np.linalg.norm(res["L"][0])
    L_fin = np.linalg.norm(res["L"][-1])
    dL = np.abs(L_fin - L0) / L0

    print("\n[3/3] Métricas de conservación y rendimiento:")
    print(f"      - Tiempo total de cómputo:  {tiempo_total:.3f} s")
    print(f"      - Tiempo por paso:          {tiempo_total / args.pasos * 1e3:.2f} ms")
    print(f"      - Error relativo máx. |ΔE|: {dE.max():.2e}  ({'CUMPLE < 1%' if dE.max() < 0.01 else 'REVISAR'})")
    print(f"      - Variación relativa de |L|: {dL:.2e}")
    print(f"      - Momento lineal máx. |P|:  {p_max:.2e} (conservación de centro de masa)")
    print("=" * 65)

if __name__ == "__main__":
    main()
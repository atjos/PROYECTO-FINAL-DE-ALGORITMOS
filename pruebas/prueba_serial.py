
from src.fisica import simular, energia, aceleraciones
import numpy as np
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def prueba_dos_cuerpos():
    print("== Prueba 1: dos cuerpos en orbita circular ==")
    # Dos masas iguales (m=1) separadas d=1, orbita circular alrededor del centro de masa.
    # Velocidad relativa = sqrt(G*M/d) con M=2  ->  cada una lleva sqrt(2)/2.
    masas = np.array([1.0, 1.0])
    pos = np.array([[0.5, 0, 0], [-0.5, 0, 0]])
    v = np.sqrt(2.0) / 2.0
    vel = np.array([[0, v, 0], [0, -v, 0]])
    eps = 1e-3
    periodo = 2 * np.pi * np.sqrt(1.0 / 2.0)   # 2*pi*sqrt(d^3/(G*M))
    dt = 0.001
    n_pasos = int(round(periodo / dt))
    res = simular(pos, vel, masas, eps, dt, n_pasos, cada=100)
    err_pos = np.linalg.norm(res["pos_final"] - pos)
    deriva_E = abs(res["E"].max() - res["E"].min()) / abs(res["E"][0])
    print(f"  Periodo teorico: {periodo:.4f}")
    print(f"  Error de posicion tras 1 periodo: {err_pos:.2e}")
    print(f"  Variacion relativa de energia:    {deriva_E:.2e}")


def prueba_cumulo():
    print("== Prueba 2: cumulo aleatorio, N=500 ==")
    rng = np.random.default_rng(42)
    N = 500
    masas = np.full(N, 1.0 / N)
    pos = rng.normal(size=(N, 3)) * 0.5
    vel = rng.normal(size=(N, 3)) * 0.3
    vel -= np.average(vel, axis=0, weights=masas)   # momento lineal total = 0
    eps, dt = 0.05, 0.005
    t0 = time.time()
    res = simular(pos, vel, masas, eps, dt, n_pasos=400, cada=20)
    seg = time.time() - t0
    E = res["E"]
    print(f"  Energia inicial: {E[0]:.6f}  final: {E[-1]:.6f}")
    print(
        f"  Variacion relativa max de E: {abs(E - E[0]).max() / abs(E[0]):.2e}")
    print(f"  |P| maximo: {np.linalg.norm(res['P'], axis=1).max():.2e}")
    print(f"  Cambio relativo de |L|: "
        f"{abs(np.linalg.norm(res['L'][-1]) - np.linalg.norm(res['L'][0])) / np.linalg.norm(res['L'][0]):.2e}")
    print(f"  Tiempo de ejecucion: {seg:.2f} s")


def prueba_escalado():
    print("== Prueba 3: tiempo de UN calculo de fuerzas vs N ==")
    rng = np.random.default_rng(0)
    for N in (250, 500, 1000, 2000):
        pos = rng.normal(size=(N, 3))
        masas = np.full(N, 1.0 / N)
        t0 = time.time()
        aceleraciones(pos, masas, 0.05)
        print(f"  N={N:5d}  {time.time() - t0:.4f} s")


if __name__ == "__main__":
    prueba_dos_cuerpos()
    prueba_cumulo()
    prueba_escalado()

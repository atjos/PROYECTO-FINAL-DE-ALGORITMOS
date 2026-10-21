"""Prueba del Segmento 3. Ejecutar desde la raiz del repo:  python pruebas/prueba_condiciones.py
Genera resultados/condiciones_iniciales.png y resultados/estabilidad_galaxia.png
"""
import sys, os
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from condiciones import crear_galaxia, crear_colision
from fisica import simular, energia

os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
EPS = 0.05


def prueba_colision():
    print("== Condiciones iniciales de la colision ==")
    pos, vel, masas, gal, nuc = crear_colision(n_por_galaxia=3000, semilla=1, eps=EPS)
    E, K, U = energia(pos, vel, masas, EPS) if len(masas) <= 3000 else (np.nan,) * 3
    print(f"  N total = {len(masas)}   masa total = {masas.sum():.3f}")
    print(f"  Momento lineal total: {np.linalg.norm((masas[:, None] * vel).sum(0)):.2e}  (debe ser ~0)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), facecolor="black")
    colores = np.where(gal == 0, "#4fc3ff", "#ff9a3c")
    for a, (i, j, tit) in zip(ax, [(0, 1, "vista x-y"), (0, 2, "vista x-z")]):
        a.set_facecolor("black")
        a.scatter(pos[~nuc, i], pos[~nuc, j], s=1.5, c=colores[~nuc], alpha=0.7)
        a.scatter(pos[nuc, i], pos[nuc, j], s=60, c="white", marker="*")
        a.set_title(tit, color="white"); a.set_aspect("equal")
        a.tick_params(colors="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(RAIZ, "resultados", "condiciones_iniciales.png"), dpi=130, facecolor="black")
    print("  Guardado: resultados/condiciones_iniciales.png")


def prueba_estabilidad():
    print("== Estabilidad de UNA galaxia aislada (N=600) ==")
    pos, vel, masas = crear_galaxia(600, eps=EPS, rng=np.random.default_rng(3))
    dt, n_pasos = 0.005, 2000                       # t = 10  (~1.6 vueltas a R = 1)
    res = simular(pos, vel, masas, EPS, dt, n_pasos, cada=100)
    E = res["E"]
    print(f"  Variacion relativa max de E: {abs(E - E[0]).max() / abs(E[0]):.2e}")

    def radios(p):                                   # distancia al nucleo (particula 0)
        return np.linalg.norm(p[1:] - p[0], axis=1)
    r0, r1 = radios(res["pos"][0]), radios(res["pos"][-1])
    print(f"  Radio mediano estrellas: inicial {np.median(r0):.3f} -> final {np.median(r1):.3f}")
    print(f"  Radio al 90% de las estrellas: inicial {np.percentile(r0, 90):.3f} -> final {np.percentile(r1, 90):.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="black")
    for a, k, tit in zip(ax[:2], (0, -1), ("t = 0", f"t = {res['t'][-1]:.1f}")):
        a.set_facecolor("black")
        a.scatter(res["pos"][k][1:, 0], res["pos"][k][1:, 1], s=2, c="#4fc3ff", alpha=0.7)
        a.scatter(*res["pos"][k][0, :2], s=60, c="white", marker="*")
        a.set_xlim(-6, 6); a.set_ylim(-6, 6); a.set_aspect("equal")
        a.set_title(tit, color="white"); a.tick_params(colors="gray")
    ax[2].set_facecolor("black")
    ax[2].plot(res["t"], (E - E[0]) / abs(E[0]), c="#ff9a3c")
    ax[2].set_title("Error relativo de energia", color="white"); ax[2].tick_params(colors="gray")
    ax[2].set_xlabel("t", color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(RAIZ, "resultados", "estabilidad_galaxia.png"), dpi=130, facecolor="black")
    print("  Guardado: resultados/estabilidad_galaxia.png")


if __name__ == "__main__":
    prueba_colision()
    prueba_estabilidad()
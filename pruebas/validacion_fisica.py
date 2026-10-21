from fisica import simular, paso_leapfrog, aceleraciones, energia
from condiciones import crear_galaxia, crear_colision
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import time
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))
matplotlib.use("Agg")

os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)

EPS = 0.05
N_POR_GALAXIA = 150        # pequeno: la version serial es lenta
# incluye el primer paso cercano (pericentro) de la colision
T_FINAL = 30.0
LISTA_DT = [0.02, 0.01, 0.005, 0.0025]
# variacion relativa maxima de energia aceptable (1 %)
CRITERIO = 0.01


# ---------------------------------------------------------------- A
def prueba_A_kepler_excentrica():
    print("== A) Dos cuerpos en orbita excentrica (e = 0.6): orden de convergencia ==")
    masas = np.array([1.0, 1.0])
    a, e = 1.0, 0.6                                   # semieje mayor y excentricidad
    M = masas.sum()
    # empezamos en el apocentro
    r_apo = a * (1 + e)
    # velocidad relativa en el apocentro
    v_rel = np.sqrt(M * (1 - e) / (a * (1 + e)))
    pos = np.array([[r_apo / 2, 0, 0], [-r_apo / 2, 0, 0]])
    vel = np.array([[0, v_rel / 2, 0], [0, -v_rel / 2, 0]])
    periodo = 2 * np.pi * np.sqrt(a**3 / M)
    eps = 1e-4
    errores = []
    for dt in (0.004, 0.002, 0.001):
        n = int(round(periodo / dt))
        dt_ajustado = periodo / n                     # un periodo exacto
        res = simular(pos, vel, masas, eps, dt_ajustado, n, cada=n)
        err = np.linalg.norm(res["pos_final"] - pos)
        errores.append(err)
        print(f"  dt = {dt:<6}  error de posicion tras 1 periodo: {err:.3e}")
    razones = [errores[i] / errores[i + 1] for i in range(len(errores) - 1)]
    print(
        f"  Razon entre errores al dividir dt entre 2: {razones[0]:.2f}, {razones[1]:.2f}  (esperado ~4 -> orden 2)")


# ---------------------------------------------------------------- B
def prueba_B_reversibilidad():
    print("== B) Reversibilidad temporal del leapfrog ==")
    pos, vel, masas = crear_galaxia(
        N_POR_GALAXIA, eps=EPS, rng=np.random.default_rng(5))
    dt, n = 0.01, 500
    acc = aceleraciones(pos, masas, EPS)
    p, v = pos.copy(), vel.copy()
    for _ in range(n):
        p, v, acc = paso_leapfrog(p, v, acc, masas, EPS, dt)
    v = -v                                            # invertimos el tiempo
    acc = aceleraciones(p, masas, EPS)
    for _ in range(n):
        p, v, acc = paso_leapfrog(p, v, acc, masas, EPS, dt)
    print(
        f"  Diferencia maxima de posicion tras ir y volver: {np.abs(p - pos).max():.2e}  (debe ser ~1e-12 o menos)")


# ---------------------------------------------------------------- C
def prueba_C_colision_vs_dt():
    print(
        f"== C) Colision completa (N = {2 * N_POR_GALAXIA + 2}) hasta t = {T_FINAL}: efecto de dt ==")
    pos, vel, masas, gal, nuc = crear_colision(
        n_por_galaxia=N_POR_GALAXIA, semilla=1, eps=EPS)
    idx = np.where(nuc)[0]
    resultados = {}
    for dt in LISTA_DT:
        n = int(round(T_FINAL / dt))
        cada = int(round(0.5 / dt))
        t0 = time.time()
        res = simular(pos, vel, masas, EPS, dt, n, cada=cada)
        E = res["E"]
        dE = np.abs(E - E[0]) / abs(E[0])
        L0, L1 = np.linalg.norm(res["L"][0]), np.linalg.norm(res["L"][-1])
        sep = np.linalg.norm(res["pos"][:, idx[0]] -
                            res["pos"][:, idx[1]], axis=1)
        resultados[dt] = dict(t=res["t"], dE=dE, sep=sep)
        print(f"  dt = {dt:<7} | max |dE/E| = {dE.max():.2e} | cambio |L| = {abs(L1 - L0) / L0:.1e} "
            f"| |P| max = {np.linalg.norm(res['P'], axis=1).max():.1e} | {time.time() - t0:.0f} s")

    validos = [dt for dt in LISTA_DT if resultados[dt]["dE"].max() < CRITERIO]
    if validos:
        print(
            f"  -> dt mas grande que cumple el criterio de {CRITERIO:.0%}: {max(validos)}")
    else:
        print("  -> ningun dt cumple el criterio; prueben valores menores en LISTA_DT")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for dt, r in resultados.items():
        ax[0].semilogy(r["t"], r["dE"] + 1e-12, label=f"dt = {dt}")
        ax[2].plot(r["t"], r["sep"], label=f"dt = {dt}")
    ax[0].axhline(CRITERIO, color="k", ls="--", lw=0.8)
    ax[0].set_title("Error relativo de energia")
    ax[0].set_xlabel("t")
    ax[0].legend()
    ax[1].loglog(LISTA_DT, [resultados[dt]["dE"].max()
                for dt in LISTA_DT], "o-")
    ax[1].axhline(CRITERIO, color="k", ls="--", lw=0.8)
    ax[1].set_title("max |dE/E| vs dt")
    ax[1].set_xlabel("dt")
    ax[2].set_title("Distancia entre nucleos")
    ax[2].set_xlabel("t")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RAIZ, "resultados", "validacion_dt.png"), dpi=130)
    print("  Guardado: resultados/validacion_dt.png")


if __name__ == "__main__":
    prueba_A_kepler_excentrica()
    prueba_B_reversibilidad()
    prueba_C_colision_vs_dt()

"""
Perfilado de la version serial: ¿que parte del paso de simulacion es la mas costosa?

Uso:
    python pruebas/perfil_serial.py            # NumPy: N = 250..2000; Numba: N = 250..16000 (total de cuerpos)
    python pruebas/perfil_serial.py --rapido   # tamanos pequenos, solo para comprobar que corre

Mide con time.perf_counter, por separado, los bloques de un paso de `fisica.simular`:
    fuerzas   : aceleraciones(pos, masas, eps)        O(N^2), una vez por paso
    energia   : energia(pos, vel, masas, eps)         O(N^2), solo cada `cada` pasos (registro)
    resto     : kick-drift-kick del leapfrog, momentos y copia de posiciones   O(N)

Cada bloque se repite REPS veces y se reporta la mediana. Con eso se calculan dos fracciones:
    % fuerzas : t_fuerzas / t_paso            (donde se va el tiempo; lo que pide el criterio 3)
    f_medida  : (t_fuerzas + t_energia/cada) / t_paso
                fraccion del paso que es trabajo O(N^2) y por tanto paralelizable. En la version
                Numba la energia tambien va en paralelo (energia_potencial_numba), asi que f_medida
                es la fraccion paralelizable REAL del codigo y se contrasta con la f efectiva del
                ajuste de Amdahl a los speedups medidos (analizar_cpu.py, f ~ 0,84).
con t_paso = t_fuerzas + t_energia/cada + t_resto.

Salida: resultados/tabla_perfil.csv (y una tabla en pantalla). Con --rapido escribe en
resultados/rapido/ para no tocar los resultados publicados.
Si Numba esta instalado, mide tambien la version Numba de 1 hilo (la referencia B del README).
"""
import csv
import os
import sys
import time

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RAIZ)

from src import fisica                       # noqa: E402
from src.condiciones import crear_colision   # noqa: E402

RAPIDO = "--rapido" in sys.argv
EPS = 0.05
DT = 0.005
CADA = 10            # igual que el registro por defecto de serial.py (pasos // 10 con 100 pasos)
REPS = 3 if RAPIDO else 9
LISTA_N_NUMPY = [250, 500] if RAPIDO else [250, 500, 1000, 2000]          # mas alla es lento y usa mucha memoria
LISTA_N_NUMBA = [250, 500] if RAPIDO else [250, 500, 1000, 2000, 4000, 8000, 16000]   # incluye los N del escalado por hilos


def mediana_tiempo(f, reps=REPS):
    """Mediana de `reps` mediciones de f() en segundos; la primera llamada se descarta (calentamiento)."""
    f()
    t = np.empty(reps)
    for k in range(reps):
        t0 = time.perf_counter()
        f()
        t[k] = time.perf_counter() - t0
    return float(np.median(t))


def datos(n_total):
    pos, vel, masas, _, _ = crear_colision(n_por_galaxia=n_total // 2 - 1, semilla=42, eps=EPS)
    return pos, vel, masas


def perfil_numpy(n_total):
    pos, vel, masas = datos(n_total)
    acc = fisica.aceleraciones(pos, masas, EPS)

    t_fuerzas = mediana_tiempo(lambda: fisica.aceleraciones(pos, masas, EPS))

    def integrador():
        v = vel + 0.5 * DT * acc          # kick
        p = pos + DT * v                  # drift
        v = v + 0.5 * DT * acc            # kick (con la misma acc: solo se mide el costo aritmetico)
        return p, v
    t_integ = mediana_tiempo(integrador)

    t_ener = mediana_tiempo(lambda: fisica.energia(pos, vel, masas, EPS))

    def otros():
        fisica.momento_lineal(vel, masas)
        fisica.momento_angular(pos, vel, masas)
        pos.copy()
    t_otros = mediana_tiempo(otros)
    return t_fuerzas, t_ener, t_integ + t_otros / CADA


def perfil_numba_1_hilo(n_total):
    from src import cpu_paralelo as cpu
    pos, vel, masas = datos(n_total)
    pos = np.ascontiguousarray(pos)
    vel = np.ascontiguousarray(vel)
    masas = np.ascontiguousarray(masas)
    acc = cpu.aceleraciones_numba_serial(pos, masas, EPS)

    t_fuerzas = mediana_tiempo(lambda: cpu.aceleraciones_numba_serial(pos, masas, EPS))

    def integrador():
        v = vel + 0.5 * DT * acc
        p = pos + DT * v
        v = v + 0.5 * DT * acc
        return p, v
    t_integ = mediana_tiempo(integrador)

    t_ener = mediana_tiempo(lambda: cpu.energia(pos, vel, masas, EPS))

    def otros():
        np.sum(masas[:, None] * vel, axis=0)
        np.sum(masas[:, None] * np.cross(pos, vel), axis=0)
        pos.copy()
    t_otros = mediana_tiempo(otros)
    return t_fuerzas, t_ener, t_integ + t_otros / CADA


def main():
    versiones = [("numpy (serial.py)", perfil_numpy, LISTA_N_NUMPY)]
    try:
        import numba  # noqa: F401
        versiones.append(("numba 1 hilo (ref. B)", perfil_numba_1_hilo, LISTA_N_NUMBA))
    except ImportError:
        print("[INFO] Numba no esta instalado: solo se perfila la version NumPy (serial.py).")

    filas = []
    print(f"Perfil de un paso de simulacion (mediana de {REPS}; energia registrada cada {CADA} pasos)\n")
    print(f"{'version':<22}{'N':>6} | {'fuerzas':>9} {'energia':>9} {'resto':>9} (ms) | "
          f"{'% fuerzas':>9} {'% energia':>9} {'% resto':>8} | {'f medida':>9}")
    for nombre, f, lista_n in versiones:
        for n in lista_n:
            tf, te, tr = f(n)               # tr: costo O(N) ya prorrateado por paso
            paso = tf + te / CADA + tr
            filas.append((nombre, n, tf, te, tr, CADA, tf / paso, (te / CADA) / paso, tr / paso,
                          (tf + te / CADA) / paso))
            print(f"{nombre:<22}{n:>6} | {tf * 1e3:>9.4f} {te * 1e3:>9.4f} {tr * 1e3:>9.5f}      | "
                  f"{100 * tf / paso:>8.2f}% {100 * (te / CADA) / paso:>8.2f}% {100 * tr / paso:>7.3f}% "
                  f"| {(tf + te / CADA) / paso:>9.5f}")

    destino = os.path.join(RAIZ, "resultados", "rapido" if RAPIDO else "")
    os.makedirs(destino, exist_ok=True)
    ruta = os.path.join(destino, "tabla_perfil.csv")
    with open(ruta, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["version", "N", "t_fuerzas_s", "t_energia_s", "t_resto_por_paso_s", "cada",
                    "frac_fuerzas", "frac_energia", "frac_resto", "f_medida"])
        w.writerows(filas)
    print(f"\nGuardado en {os.path.relpath(ruta, RAIZ)}")
    print("f medida = fraccion O(N^2) del paso (paralelizable); compararla con la f del ajuste de Amdahl.")


if __name__ == "__main__":
    main()

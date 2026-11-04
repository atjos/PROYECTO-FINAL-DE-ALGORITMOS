"""
Benchmark de CPU (v3): NumPy vs Numba (1 hilo) vs Numba paralelo.

Uso (desde la raiz del proyecto, o desde cualquier carpeta):
    python pruebas/benchmark_cpu.py                 # 4 corridas completas (por defecto)
    python pruebas/benchmark_cpu.py --corridas 5    # otro numero de corridas
    python pruebas/benchmark_cpu.py --pausa 30      # segundos de descanso entre corridas (def. 20)
    python pruebas/benchmark_cpu.py --rapido        # prueba corta; escribe en benchmarks/rapido/ y resultados/rapido/

CADA corrida escribe su PROPIO archivo, con fecha y numero de corrida:
    benchmarks/resultados_cpu_AAAAMMDD_HHMMSS_c1.csv, ..._c2.csv, ...
Al terminar TODAS las corridas, los resultados_cpu*.csv anteriores se MUEVEN (no se borran) a
benchmarks/archivo/<fecha>/, de modo que benchmarks/ contiene siempre un solo lote y el analisis
(python pruebas/analizar_cpu.py) no mezcla mediciones de sesiones distintas.
Con --rapido nada de lo anterior se toca: todo va a las subcarpetas rapido/.

Otras salidas:
    benchmarks/verificacion_cpu.csv    error serial vs paralelo para varios N (se hace una vez)
    resultados/hardware.txt            hardware y versiones usadas
    resultados/benchmark_cpu.png       figura de la ULTIMA corrida

Cada tiempo se reporta como MEDIANA de las repeticiones, con su dispersion (IQR y min-max).
La primera llamada (compilacion JIT) se descarta siempre.
"""
import csv
import glob
import os
import platform
import shutil
import sys
import time

RAIZ = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RAIZ)          # permite "from src import ..." desde cualquier carpeta

import matplotlib                       # noqa: E402
import numba                            # noqa: E402
import numpy as np                      # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402

from src import cpu_paralelo as cpu     # noqa: E402
from src import fisica                  # noqa: E402
from src.condiciones import crear_colision  # noqa: E402

VERSION = "benchmark_cpu v3 (un CSV por corrida)"
RAPIDO = "--rapido" in sys.argv

# La prueba rapida NUNCA escribe sobre los resultados publicados: va a subcarpetas rapido/.
DIR_BENCH = os.path.join(RAIZ, "benchmarks", "rapido") if RAPIDO else os.path.join(RAIZ, "benchmarks")
DIR_RES = os.path.join(RAIZ, "resultados", "rapido") if RAPIDO else os.path.join(RAIZ, "resultados")
os.makedirs(DIR_RES, exist_ok=True)
os.makedirs(DIR_BENCH, exist_ok=True)


def _arg(nombre, defecto, tipo=int):
    if nombre in sys.argv:
        i = sys.argv.index(nombre)
        try:
            return tipo(sys.argv[i + 1])
        except (IndexError, ValueError):
            sys.exit(f"Uso: {nombre} <valor>")
    return defecto


N_CORRIDAS = _arg("--corridas", 1 if RAPIDO else 4)
PAUSA_S = _arg("--pausa", 0 if RAPIDO else 20, float)
SELLO = time.strftime("%Y%m%d_%H%M%S")      # identifica este lote de corridas
EPS = 0.05
TOL = 1e-10          # tolerancia de verificacion: el error esperado en float64 es ~1e-13..1e-15
MAX_HILOS = numba.config.NUMBA_NUM_THREADS

if RAPIDO:
    LISTA_N, N_NUMPY_MAX, LISTA_N_HILOS, REPS = [250, 500, 1000], 1000, [1000], 3
    REPS_ESC = 5
    LISTA_N_VERIF = [250, 1000]
else:
    LISTA_N = [250, 500, 1000, 2000, 4000, 8000, 16000]
    N_NUMPY_MAX = 2000
    LISTA_N_HILOS = [2000, 16000]       # al menos dos tamanos para el escalado por hilos
    LISTA_N_VERIF = [250, 1000, 2000, 4000]
    REPS = 7
    REPS_ESC = 15       # el escalado por hilos se repite mas veces (tiempos cortos = mas ruido)

LISTA_HILOS = sorted({h for h in (1, 2, 4, 6, 8, 12, 16) if h <= MAX_HILOS} | {MAX_HILOS})


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def medir(f, *args, reps=REPS):
    """Devuelve el arreglo con los 'reps' tiempos (s). La 1a llamada (JIT) no se cuenta."""
    f(*args)
    tiempos = np.empty(reps)
    for k in range(reps):
        t0 = time.perf_counter()
        f(*args)
        tiempos[k] = time.perf_counter() - t0
    return tiempos


def medir_intercalado(configs, reps):
    """
    configs: dict nombre -> (preparar, ejecutar), dos funciones sin argumentos.
    'preparar' (p. ej. fijar el numero de hilos) se llama ANTES de cronometrar y no cuenta en el tiempo.
    En cada ronda se mide UNA vez cada configuracion (en orden), y se repite 'reps' rondas, para que la
    deriva termica o de frecuencia del portatil afecte a todas por igual. La ronda 0 (JIT) se descarta.
    """
    tiempos = {k: [] for k in configs}
    for ronda in range(reps + 1):
        for k, (preparar, ejecutar) in configs.items():
            preparar()
            t0 = time.perf_counter()
            ejecutar()
            dt = time.perf_counter() - t0
            if ronda > 0:
                tiempos[k].append(dt)
    return {k: np.array(v) for k, v in tiempos.items()}


def resumen(t):
    """(mediana, rango intercuartilico, minimo, maximo)."""
    q1, q3 = np.percentile(t, [25, 75])
    return float(np.median(t)), float(q3 - q1), float(t.min()), float(t.max())


def amdahl_f(speedup, p):
    """Fraccion paralelizable f que explicaria S con p hilos: S = 1 / ((1-f) + f/p)."""
    if p <= 1:
        return np.nan
    return (1.0 - 1.0 / speedup) / (1.0 - 1.0 / p)


def datos(N, semilla=0):
    rng = np.random.default_rng(semilla)
    return rng.normal(size=(N, 3)), np.full(N, 1.0 / N)


def _powershell(cmd):
    import subprocess
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:
        return ""


def nucleos_fisicos():
    """Nucleos fisicos: psutil si esta; si no, en Windows se consulta con PowerShell. None si no se pudo."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    if platform.system() == "Windows":
        txt = _powershell("(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
        if txt.isdigit():
            return int(txt)
    return None


def nombre_cpu():
    if platform.system() == "Windows":
        txt = _powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        if txt:
            return txt
    return platform.processor() or "desconocido"


def info_hardware():
    fis = nucleos_fisicos()
    lineas = [
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Script: {VERSION}",
        f"Sistema: {platform.platform()}",
        f"Procesador: {nombre_cpu()}",
        f"Python: {platform.python_version()}",
        f"NumPy: {np.__version__}   Numba: {numba.__version__}",
        f"Nucleos fisicos: {fis if fis else 'no detectado (anotalo a mano)'}",
        f"Hilos logicos (os.cpu_count): {os.cpu_count()}",
        f"Hilos usados por Numba (maximo): {MAX_HILOS}",
    ]
    try:
        import psutil
        lineas.append(f"RAM total: {psutil.virtual_memory().total / 2**30:.1f} GiB")
    except ImportError:
        pass
    return lineas


# ----------------------------------------------------------------------------
# 1 y 2. Verificacion serial vs paralelo
# ----------------------------------------------------------------------------
def archivar_lotes_previos(nuevos):
    """Mueve a benchmarks/archivo/<SELLO>/ los resultados_cpu*.csv que NO son de esta sesion.
    Se llama solo cuando todas las corridas terminaron bien: si algo falla antes, los datos
    anteriores siguen donde estaban. Mueve, nunca borra."""
    previos = [r for r in sorted(glob.glob(os.path.join(DIR_BENCH, "resultados_cpu*.csv")))
               if os.path.abspath(r) not in {os.path.abspath(n) for n in nuevos}]
    if not previos:
        return
    destino = os.path.join(DIR_BENCH, "archivo", SELLO)
    os.makedirs(destino, exist_ok=True)
    for r in previos:
        shutil.move(r, os.path.join(destino, os.path.basename(r)))
    print(f"\n  {len(previos)} CSV de sesiones anteriores movidos a {os.path.relpath(destino, RAIZ)}/"
          " (benchmarks/ queda solo con este lote)")


def prueba_correccion():
    print(f"--- 1. Verificacion de aceleraciones vs referencia NumPy (tolerancia {TOL:.0e}) ---")
    filas = []
    print(f"  {'N':>6} | {'implementacion':>15} | {'error rel. max':>14} | resultado")
    for N in LISTA_N_VERIF:
        pos, masas = datos(N)
        ref = fisica.aceleraciones(pos, masas, EPS)
        for nombre, f in (("numba 1 hilo", cpu.aceleraciones_numba_serial),
                          ("numba paralelo", cpu.aceleraciones_numba)):
            a = f(pos, masas, EPS)
            err = float(np.abs(a - ref).max() / np.abs(ref).max())
            ok = err < TOL
            filas.append((N, nombre, err, ok))
            print(f"  {N:6d} | {nombre:>15} | {err:14.2e} | {'PASA' if ok else 'NO PASA'}")
    with open(os.path.join(DIR_BENCH, "verificacion_cpu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "implementacion", "error_rel_max", "pasa_tolerancia"])
        w.writerows(filas)

    pos, masas = datos(1000)
    E_ref = fisica.energia(pos, np.zeros_like(pos), masas, EPS)[2]
    E_nb = cpu.energia(pos, np.zeros_like(pos), masas, EPS)[2]
    print(f"  Energia potencial (N=1000): NumPy {E_ref:.10f}   Numba {E_nb:.10f}   "
          f"dif. relativa {abs(E_nb / E_ref - 1):.1e}")

    print("--- 2. Verificacion dinamica: 400 pasos de colision (N = 302) ---")
    p, v, m, _, _ = crear_colision(n_por_galaxia=150, semilla=1, eps=EPS)
    r_ref = fisica.simular(p, v, m, EPS, 0.005, 400, cada=100)
    r_nb = cpu.simular(p, v, m, EPS, 0.005, 400, cada=100)
    print(f"  Diferencia maxima de posicion tras 400 pasos: "
          f"{np.abs(r_ref['pos_final'] - r_nb['pos_final']).max():.2e}")
    print(f"  Variacion de energia: NumPy {abs(r_ref['E'][-1] / r_ref['E'][0] - 1):.2e}   "
          f"Numba {abs(r_nb['E'][-1] / r_nb['E'][0] - 1):.2e}")


# ----------------------------------------------------------------------------
# 3 y 4. Rendimiento
# ----------------------------------------------------------------------------
def prueba_rendimiento(corrida, ruta_csv):
    filas = []      # (implementacion, N, hilos, rep, tiempo_s)

    def guardar(nombre, N, h, tiempos):
        for k, t in enumerate(tiempos):
            filas.append((nombre, N, h, k, float(t)))

    # ---- 3. Tiempo vs N ----
    print(f"=== CORRIDA {corrida}/{N_CORRIDAS} ===")
    print(f"--- 3. Tiempo de calculo de fuerzas vs N (mediana de {REPS}; hilos max = {MAX_HILOS}) ---")
    numba.set_num_threads(MAX_HILOS)
    res_np, res_s, res_p = {}, {}, {}
    print(f"  {'N':>6} | {'NumPy (ms)':>11} | {'Numba 1h (ms)':>13} | {'Numba ' + str(MAX_HILOS) + 'h (ms)':>14} "
          f"| {'IQR par. (ms)':>13} | {'S 1h->' + str(MAX_HILOS) + 'h':>9} | {'Mint/s par.':>11}")
    for N in LISTA_N:
        pos, masas = datos(N)
        t_s = medir(cpu.aceleraciones_numba_serial, pos, masas, EPS)
        t_p = medir(cpu.aceleraciones_numba, pos, masas, EPS)
        guardar("numba_1_hilo", N, 1, t_s)
        guardar("numba_paralelo", N, MAX_HILOS, t_p)
        res_s[N], res_p[N] = resumen(t_s), resumen(t_p)
        txt_np = f"{'-':>11}"
        if N <= N_NUMPY_MAX:
            t_np = medir(fisica.aceleraciones, pos, masas, EPS, reps=min(REPS, 3))
            guardar("numpy", N, 1, t_np)
            res_np[N] = resumen(t_np)
            txt_np = f"{res_np[N][0] * 1e3:11.3f}"
        print(f"  {N:6d} | {txt_np} | {res_s[N][0] * 1e3:13.3f} | {res_p[N][0] * 1e3:14.3f} "
              f"| {res_p[N][1] * 1e3:13.3f} | {res_s[N][0] / res_p[N][0]:9.2f} | {N * N / res_p[N][0] / 1e6:11.0f}")
    if res_np:
        n_ref = max(res_np)
        print(f"  Numba paralelo vs NumPy (N = {n_ref}): {res_np[n_ref][0] / res_p[n_ref][0]:.1f}x "
              f"(ojo: mezcla compilacion + paralelismo; el speedup de paralelismo puro es el de la seccion 4)")

    # ---- 4. Escalabilidad por hilos, para varios N ----
    # Referencia T1 = MISMO codigo paralelo ejecutado con 1 hilo (definicion estandar de S_p = T1/Tp).
    # El Numba "serial" (sin prange) se mide en las mismas rondas, solo como dato aparte.
    escal = {}      # N -> lista de (h, mediana, iqr, speedup, eficiencia, f)
    for N in LISTA_N_HILOS:
        print(f"--- 4. Escalabilidad vs numero de hilos (N = {N}, {REPS_ESC} rondas intercaladas) ---")
        pos, masas = datos(N)

        def lanzar(h):
            return (lambda: numba.set_num_threads(h),
                    lambda: cpu.aceleraciones_numba(pos, masas, EPS))

        configs = {"serial": (lambda: None,
                              lambda: cpu.aceleraciones_numba_serial(pos, masas, EPS))}
        configs.update({h: lanzar(h) for h in LISTA_HILOS})
        tiempos = medir_intercalado(configs, REPS_ESC)
        numba.set_num_threads(MAX_HILOS)

        guardar("numba_serial_base", N, 1, tiempos["serial"])
        for h in LISTA_HILOS:
            guardar("numba_paralelo_hilos", N, h, tiempos[h])

        T1 = float(np.median(tiempos[1]))
        t_ser = float(np.median(tiempos["serial"]))
        print(f"  Referencia T1 (codigo paralelo con 1 hilo): {T1 * 1e3:.3f} ms")
        print(f"  Numba compilado sin prange: {t_ser * 1e3:.3f} ms  (T_serial / T1 = {t_ser / T1:.2f})")
        if N in res_s and abs(t_ser / res_s[N][0] - 1) > 0.25:
            print(f"  AVISO: el serial difiere {abs(t_ser / res_s[N][0] - 1):.0%} de la seccion 3 "
                  f"({res_s[N][0] * 1e3:.3f} ms): hubo deriva o ruido; repite con el equipo en reposo.")
        print(f"  {'hilos':>5} | {'mediana (ms)':>12} | {'IQR (ms)':>9} | {'min-max (ms)':>17} | "
              f"{'speedup':>7} | {'efic.':>6} | {'f Amdahl':>8}")
        escal[N] = []
        for h in LISTA_HILOS:
            med, iqr, tmin, tmax = resumen(tiempos[h])
            sp = T1 / med
            ef = sp / h
            f_a = amdahl_f(sp, h)
            escal[N].append((h, med, iqr, sp, ef, f_a))
            txt_f = f"{f_a:8.3f}" if not np.isnan(f_a) else f"{'-':>8}"
            print(f"  {h:5d} | {med * 1e3:12.3f} | {iqr * 1e3:9.3f} | {tmin * 1e3:7.3f}-{tmax * 1e3:7.3f} | "
                  f"{sp:7.2f} | {ef:6.0%} | {txt_f}")

    # ---- CSV con todas las repeticiones ----
    with open(ruta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementacion", "N", "hilos", "rep", "tiempo_s"])
        w.writerows(filas)

    # ---- Figura: tiempo vs N | speedup | eficiencia ----
    fisicos = nucleos_fisicos()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    def serie(res, etiqueta, marcador):
        Ns = list(res)
        med = np.array([res[n][0] for n in Ns])
        lo = med - np.array([res[n][2] for n in Ns])
        hi = np.array([res[n][3] for n in Ns]) - med
        ax[0].errorbar(Ns, med, yerr=[lo, hi], fmt=marcador + "-", capsize=3, label=etiqueta)

    serie(res_np, "NumPy (serial)", "o")
    serie(res_s, "Numba, 1 hilo", "s")
    serie(res_p, f"Numba, {MAX_HILOS} hilos", "^")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Numero de particulas (N)")
    ax[0].set_ylabel("Tiempo por calculo de fuerzas (s)")
    ax[0].set_title(f"Tiempo vs. N (mediana y min-max de {REPS} rep.)")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")

    ax[1].plot(LISTA_HILOS, LISTA_HILOS, "k--", lw=0.8, label="ideal")
    for N, datos_n in escal.items():
        ax[1].plot([d[0] for d in datos_n], [d[3] for d in datos_n], "o-", label=f"medido, N = {N}")
        ax[2].plot([d[0] for d in datos_n], [d[4] for d in datos_n], "o-", label=f"medido, N = {N}")
    ax[2].axhline(1.0, color="k", ls="--", lw=0.8, label="ideal")
    for a in ax[1:]:
        if fisicos:
            a.axvline(fisicos, color="gray", ls=":", lw=1, label=f"{fisicos} nucleos fisicos")
        a.set_xlabel("Numero de hilos")
        a.grid(alpha=0.3)
        a.legend()
    ax[1].set_ylabel("Speedup  S = T1 / Tp  (T1: 1 hilo)")
    ax[1].set_title("Escalabilidad multihilo")
    ax[2].set_ylabel("Eficiencia  E = S / p")
    ax[2].set_ylim(0, 1.1)
    ax[2].set_title("Eficiencia paralela")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_RES, "benchmark_cpu.png"), dpi=130)
    print(f"  Guardado: {os.path.relpath(os.path.join(DIR_RES, 'benchmark_cpu.png'), RAIZ)} y {os.path.relpath(ruta_csv, RAIZ)}")


if __name__ == "__main__":
    hw = info_hardware()
    print(VERSION)
    print("\n".join(hw) + "\n")
    with open(os.path.join(DIR_RES, "hardware.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(hw) + "\n")
    prueba_correccion()
    generados = []
    for k in range(1, N_CORRIDAS + 1):
        ruta = os.path.join(DIR_BENCH, f"resultados_cpu_{SELLO}_c{k}.csv")
        prueba_rendimiento(k, ruta)
        generados.append(ruta)
        if k < N_CORRIDAS and PAUSA_S > 0:
            print(f"\n  (pausa de {PAUSA_S:.0f} s para que la CPU se enfrie antes de la siguiente corrida)\n")
            time.sleep(PAUSA_S)
    if not RAPIDO:
        archivar_lotes_previos(generados)
    print("\n=== ARCHIVOS GENERADOS ===")
    for r in generados:
        print("  ", os.path.relpath(r, RAIZ), "OK" if os.path.exists(r) else "NO SE CREO")
    if RAPIDO:
        print("Prueba rapida: los resultados publicados no se tocaron (todo quedo en las carpetas rapido/).")
    else:
        print("Ahora ejecuta:  python pruebas/analizar_cpu.py")
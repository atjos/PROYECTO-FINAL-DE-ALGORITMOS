"""
Benchmark de GPU (v3): kernel CUDA vs Numba serial vs Numba paralelo, con transferencias separadas.

Uso (desde la raiz del proyecto o desde cualquier carpeta):
    python pruebas/benchmark_gpu.py             # corrida completa
    python pruebas/benchmark_gpu.py --rapido    # prueba corta, solo para ver que corre

Que mide (todo con time.perf_counter; la 1a llamada de cada configuracion -compilacion JIT- se descarta):
    - gpu_kernel : solo el kernel de fuerzas, con los datos ya en la GPU (caso de la simulacion real,
                donde posiciones y velocidades se quedan en la GPU entre pasos).
    - gpu_h2d    : copia host -> device de las posiciones y masas (incluye reservar memoria en la GPU).
    - gpu_d2h    : copia device -> host de las aceleraciones.
    - cpu_serial : Numba compilado sin paralelismo (1 hilo).
    - cpu_paralelo: Numba con prange y todos los hilos.
    Cada tiempo se reporta como MEDIANA de las repeticiones, con IQR y min-max.

Cambios de la v3 respecto a la v2:
    - Fase GPU primero, con los kernels de TODOS los N medidos INTERCALADOS por rondas y sin carga de CPU
    en medio; la fase CPU va despues, tras una pausa. Asi el throttling termico/de potencia de un
    portatil (CPU y GPU comparten presupuesto) no distorsiona un solo N, como paso con N = 16000.
    - Chequeo automatico: avisa si el kernel de N_TPB difiere >25 % entre el barrido principal y la
    prueba de TPB (misma configuracion, deben coincidir).
    - Hardware: nombre real de la CPU y nucleos fisicos (antes imprimia "Family 6 Model 141").
    - Se silencia el aviso de baja ocupacion de Numba (es esperable con N pequeno; ver el informe).

Verificacion rapida (sin repetir el benchmark; no modifica resultados_gpu.csv):
    python pruebas/benchmark_gpu.py --solo-verificacion

Salidas:
    benchmarks/verificacion_gpu.csv    error GPU (float32) vs CPU (float64): aceleraciones para varios N,
                                       dinamica y conservacion de energia (prueba, N, magnitud, valor, tolerancia, pasa)
    benchmarks/resultados_gpu.csv      todas las repeticiones (implementacion, N, hilos_o_tpb, rep, tiempo_s)
    resultados/hardware_gpu.txt        GPU, CPU y versiones usadas
    resultados/benchmark_gpu.png       tiempo vs N, speedup, desglose de tiempo y sensibilidad a TPB
"""
import csv
import os
import platform
import sys
import time
import warnings

RAIZ = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
# permite "from src import ..." desde cualquier carpeta
sys.path.insert(0, RAIZ)

import matplotlib                       # noqa: E402
import numba                            # noqa: E402
import numpy as np                      # noqa: E402
from numba import cuda                  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402

from src import cpu_paralelo as cpu     # noqa: E402
from src import fisica                  # noqa: E402
from src import gpu_cuda as gpu         # noqa: E402
from src.condiciones import crear_colision  # noqa: E402

# Con N pequeno hay pocos bloques para los SM de la GPU: el aviso es cierto pero esperado.
warnings.filterwarnings("ignore", message=".*under-utilization.*")

os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
os.makedirs(os.path.join(RAIZ, "benchmarks"), exist_ok=True)

RAPIDO = "--rapido" in sys.argv
EPS = 0.05
N_CORRECCION = 1000
N_POR_GAL_ENERGIA = 1000
PASOS_ENERGIA = 6000
N_TPB = 16000
# Verificacion GPU (float32) frente a la referencia en float64: varios N, resultado guardado en CSV.
# Referencia = Numba serial en float64, que ya coincide con NumPy a ~1e-16 (benchmarks/verificacion_cpu.csv).
LISTA_N_VERIF = [250, 1000, 2000, 4000, 8000, 16000]
# Tolerancia de aceleraciones en float32: el epsilon de maquina es ~6e-8 y cada aceleracion suma N terminos,
# asi que se espera un error relativo del orden de 1e-7 a 1e-6; 1e-5 deja margen sin ocultar errores reales.
TOL_GPU = 1e-5
TOL_ENERGIA = 0.01      # criterio del proyecto: |dE/E| < 1 %
RUTA_VERIF = os.path.join(RAIZ, "benchmarks", "verificacion_gpu.csv")
LISTA_TPB = [64, 128, 256, 512]
# el serial a N mayor tarda demasiado; la CPU se mide solo hasta aqui
N_CPU_MAX = 16000
MAX_HILOS = numba.config.NUMBA_NUM_THREADS
# diferencia relativa maxima aceptada entre dos mediciones de la misma config.
TOL_AVISO = 0.25

if RAPIDO:
    LISTA_N, PASOS_ENERGIA = [1000, 4000, 16000], 1000
    REPS_GPU, REPS_CPU = 5, 3
    PAUSA_S = 0
else:
    LISTA_N = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
    # los tiempos de GPU son cortos (sub-ms): mas repeticiones = menos ruido
    REPS_GPU = 15
    REPS_CPU = 5        # minimo pedido por la rubrica
    PAUSA_S = 20        # descanso entre la fase GPU y la fase CPU


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def datos(N, semilla=0):
    rng = np.random.default_rng(semilla)
    return rng.normal(size=(N, 3)), np.full(N, 1.0 / N)


def resumen(t):
    """(mediana, rango intercuartilico, minimo, maximo)."""
    q1, q3 = np.percentile(t, [25, 75])
    return float(np.median(t)), float(q3 - q1), float(t.min()), float(t.max())


def medir(f, *args, reps):
    """Arreglo con los 'reps' tiempos (s) de f(*args). La 1a llamada (JIT) no se cuenta."""
    f(*args)
    t = np.empty(reps)
    for k in range(reps):
        t0 = time.perf_counter()
        f(*args)
        t[k] = time.perf_counter() - t0
    return t


def medir_gpu(f, reps):
    """Igual, pero sincronizando antes y despues: los lanzamientos de kernels son asincronos."""
    f()
    cuda.synchronize()
    t = np.empty(reps)
    for k in range(reps):
        cuda.synchronize()
        t0 = time.perf_counter()
        f()
        cuda.synchronize()
        t[k] = time.perf_counter() - t0
    return t


def medir_transferencias(N, reps):
    """
    Tiempos de copia host->device (posiciones+masas, float32 empaquetado en (N,4)) y device->host
    (aceleraciones, (N,3) float32). El empaquetado float64->float32 en CPU no se cuenta.
    La memoria es pageable (la de NumPy por defecto): es lo que usa gpu.aceleraciones().
    """
    pos, masas = datos(N)
    pos4 = gpu.empaquetar(pos, masas)
    d_acc = cuda.device_array((N, 3), dtype=np.float32)
    t_h2d, t_d2h = [], []
    # la ronda 0 se descarta (inicializacion del contexto)
    for ronda in range(reps + 1):
        cuda.synchronize()
        t0 = time.perf_counter()
        d_pos4 = cuda.to_device(pos4)
        cuda.synchronize()
        a = time.perf_counter() - t0
        t0 = time.perf_counter()
        d_acc.copy_to_host()
        cuda.synchronize()
        b = time.perf_counter() - t0
        if ronda > 0:
            t_h2d.append(a)
            t_d2h.append(b)
        del d_pos4
    return np.array(t_h2d), np.array(t_d2h)


def lanzador(N, tpb=gpu.TPB_POR_DEFECTO):
    """Prepara arrays en la GPU y devuelve una funcion que lanza SOLO el kernel de fuerzas."""
    pos, masas = datos(N)
    k_acc = gpu.construir_kernels(tpb)[0]
    d_pos4 = cuda.to_device(gpu.empaquetar(pos, masas))
    d_acc = cuda.device_array((N, 3), dtype=np.float32)
    eps2 = np.float32(EPS * EPS)
    bloques = (N + tpb - 1) // tpb
    return lambda: k_acc[bloques, tpb](d_pos4, d_acc, N, eps2)


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
        txt = _powershell(
            "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
        if txt.isdigit():
            return int(txt)
    return None


def nombre_cpu():
    if platform.system() == "Windows":
        txt = _powershell(
            "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        if txt:
            return txt
    return platform.processor() or "desconocido"


def info_hardware_gpu():
    dev = cuda.get_current_device()
    nombre = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
    lineas = [
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Sistema: {platform.platform()}",
        f"GPU: {nombre}",
    ]
    try:
        lineas.append(
            "Capacidad de computo: %d.%d" % tuple(dev.compute_capability))
    except Exception:
        pass
    try:
        lineas.append(f"Multiprocesadores (SM): {dev.MULTIPROCESSOR_COUNT}")
    except Exception:
        pass
    try:
        lineas.append(
            f"Memoria de la GPU: {cuda.current_context().get_memory_info().total / 2**30:.1f} GiB")
    except Exception:
        pass
    try:
        lineas.append("CUDA runtime: %d.%d" %
                    tuple(cuda.runtime.get_version()))
    except Exception:
        pass
    fis = nucleos_fisicos()
    lineas += [
        f"Python: {platform.python_version()}   NumPy: {np.__version__}   Numba: {numba.__version__}",
        f"CPU de comparacion: {nombre_cpu()}",
        f"Nucleos fisicos: {fis if fis else 'no detectado (anotalo a mano)'}   "
        f"hilos logicos: {os.cpu_count()}   hilos Numba: {MAX_HILOS}",
        "Precision: GPU en float32, CPU en float64 (la comparacion de tiempos mezcla ambas).",
    ]
    return nombre, lineas


# ----------------------------------------------------------------------------
# 1 a 3. Correccion
# ----------------------------------------------------------------------------
def _guardar_verificacion(filas, anexar):
    """Escribe (o agrega a) benchmarks/verificacion_gpu.csv."""
    modo = "a" if anexar else "w"
    with open(RUTA_VERIF, modo, newline="") as f:
        w = csv.writer(f)
        if not anexar:
            w.writerow(["prueba", "N", "magnitud", "valor", "tolerancia", "pasa_tolerancia"])
        w.writerows(filas)


def prueba_correccion():
    filas = []
    print(f"--- 1. Verificación de aceleraciones: GPU (float32) vs Numba serial (float64), "
          f"tolerancia {TOL_GPU:.0e} ---")
    for N in LISTA_N_VERIF:
        pos, masas = datos(N)
        ref = cpu.aceleraciones_numba_serial(pos, masas, EPS)
        a = gpu.aceleraciones(pos, masas, EPS)
        err = float(np.abs(a - ref).max() / np.abs(ref).max())
        ok = err < TOL_GPU
        filas.append(("aceleraciones", N, "error_rel_max", err, TOL_GPU, ok))
        print(f"  N = {N:6d}   error relativo maximo en a: {err:.2e}   {'OK' if ok else 'NO PASA'}")

    pos, masas = datos(N_CORRECCION)
    r = gpu.simular(pos, np.zeros_like(pos), masas,
                    EPS, 0.01, 0, guardar_pos=False)
    U_ref = fisica.energia(pos, np.zeros_like(pos), masas, EPS)[2]
    err_U = abs(r["E"][0] - U_ref) / abs(U_ref)
    filas.append(("energia_potencial", N_CORRECCION, "error_rel", float(err_U), TOL_GPU, bool(err_U < TOL_GPU)))
    print(f"  Energia potencial (N = {N_CORRECCION}): NumPy {U_ref:.6f}   GPU {r['E'][0]:.6f}   error relativo {err_U:.2e}")

    print("--- 2. Dinámica orbital: 400 pasos de colisión (N = 302), GPU vs CPU float64 ---")
    p, v, m, _, _ = crear_colision(n_por_galaxia=150, semilla=1, eps=EPS)
    r_cpu = cpu.simular(p, v, m, EPS, 0.005, 400, cada=100, guardar_pos=False)
    r_gpu = gpu.simular(p, v, m, EPS, 0.005, 400, cada=100, guardar_pos=False)
    dpos = float(np.abs(r_cpu["pos_final"] - r_gpu["pos_final"]).max())
    dE_cpu = float(abs(r_cpu["E"][-1] / r_cpu["E"][0] - 1))
    dE_gpu = float(abs(r_gpu["E"][-1] / r_gpu["E"][0] - 1))
    # La diferencia de posiciones es informativa (el caos amplifica el error de float32): no tiene tolerancia.
    filas.append(("dinamica_400_pasos", len(m), "dif_max_posicion", dpos, "", "informativo"))
    filas.append(("dinamica_400_pasos", len(m), "variacion_energia_cpu", dE_cpu, TOL_ENERGIA, bool(dE_cpu < TOL_ENERGIA)))
    filas.append(("dinamica_400_pasos", len(m), "variacion_energia_gpu", dE_gpu, TOL_ENERGIA, bool(dE_gpu < TOL_ENERGIA)))
    print(f"  Diferencia maxima de posicion tras 400 pasos: {dpos:.2e}"
          "  (el error de float32 se amplifica por la dinamica caotica)")
    print(f"  Variacion de energia: CPU {dE_cpu:.2e}   GPU {dE_gpu:.2e}")
    _guardar_verificacion(filas, anexar=False)


def prueba_energia_float32():
    n = 2 * N_POR_GAL_ENERGIA + 2
    print(
        f"--- 3. Conservación de energía en float32: N = {n}, {PASOS_ENERGIA} pasos ---")
    p, v, m, _, _ = crear_colision(
        n_por_galaxia=N_POR_GAL_ENERGIA, semilla=1, eps=EPS)
    t0 = time.perf_counter()
    r = gpu.simular(p, v, m, EPS, 0.005, PASOS_ENERGIA,
                    cada=100, guardar_pos=False)
    seg = time.perf_counter() - t0
    dE = np.abs(r["E"] - r["E"][0]) / abs(r["E"][0])
    L0, L1 = np.linalg.norm(r["L"][0]), np.linalg.norm(r["L"][-1])
    dL = float(abs(L1 - L0) / L0)
    pmax = float(np.linalg.norm(r["P"], axis=1).max())
    print(f"  max |dE/E| = {dE.max():.2e}   cambio relativo |L| = {dL:.1e}   "
          f"|P| max = {pmax:.1e}   ({seg:.1f} s en total)")
    print(
        f"  Criterio del 1 %: {'CUMPLE' if dE.max() < TOL_ENERGIA else 'NO CUMPLE'}")
    _guardar_verificacion([
        (f"energia_float32_{PASOS_ENERGIA}_pasos", n, "max_dE_sobre_E", float(dE.max()), TOL_ENERGIA, bool(dE.max() < TOL_ENERGIA)),
        (f"energia_float32_{PASOS_ENERGIA}_pasos", n, "cambio_rel_L", dL, "", "informativo"),
        (f"energia_float32_{PASOS_ENERGIA}_pasos", n, "P_max", pmax, "", "informativo"),
    ], anexar=True)


# ----------------------------------------------------------------------------
# 4 y 5. Rendimiento
# ----------------------------------------------------------------------------
def prueba_rendimiento(dev):
    filas = []      # (implementacion, N, hilos_o_tpb, rep, tiempo_s)

    def guardar(nombre, N, h, tiempos):
        for k, t in enumerate(tiempos):
            filas.append((nombre, N, h, k, float(t)))

    numba.set_num_threads(MAX_HILOS)
    tpb0 = gpu.TPB_POR_DEFECTO
    print(f"--- 4. Tiempo de cálculo vs N (GPU: {dev}, TPB = {tpb0}) ---")
    print(f"  Mediana de {REPS_GPU} rep. en GPU y {REPS_CPU} en CPU; IQR entre parentesis; "
        f"CPU: 1 hilo y {MAX_HILOS} hilos (solo hasta N = {N_CPU_MAX}).")
    print("  Orden: (a) kernels de todos los N intercalados por rondas, (b) transferencias, "
        "(c) pausa, (d) CPU.")
    res = {k: {} for k in ("serial", "paralelo", "kernel", "h2d", "d2h")}

    # --- (a) Kernels de TODOS los N, intercalados por rondas, sin carga de CPU en medio ---
    lanz = {N: lanzador(N, tpb0) for N in LISTA_N}
    for N in LISTA_N:                       # calentamiento: JIT y reloj de la GPU
        lanz[N]()
        cuda.synchronize()
    t_kern = {N: [] for N in LISTA_N}
    for _ in range(REPS_GPU):
        for N in LISTA_N:
            cuda.synchronize()
            t0 = time.perf_counter()
            lanz[N]()
            cuda.synchronize()
            t_kern[N].append(time.perf_counter() - t0)

    # --- (b) Transferencias ---
    for N in LISTA_N:
        t_k = np.array(t_kern[N])
        guardar("gpu_kernel", N, tpb0, t_k)
        res["kernel"][N] = resumen(t_k)
        t_h, t_d = medir_transferencias(N, REPS_GPU)
        guardar("gpu_h2d", N, tpb0, t_h)
        guardar("gpu_d2h", N, tpb0, t_d)
        res["h2d"][N], res["d2h"][N] = resumen(t_h), resumen(t_d)

    # --- (c) Pausa para que el equipo se enfrie, y (d) fase CPU ---
    if PAUSA_S > 0:
        print(f"  (pausa de {PAUSA_S} s antes de medir la CPU)")
        time.sleep(PAUSA_S)
    for N in LISTA_N:
        if N <= N_CPU_MAX:
            pos, masas = datos(N)
            t_s = medir(cpu.aceleraciones_numba_serial,
                        pos, masas, EPS, reps=REPS_CPU)
            t_p = medir(cpu.aceleraciones_numba, pos,
                        masas, EPS, reps=REPS_CPU)
            guardar("cpu_serial", N, 1, t_s)
            guardar("cpu_paralelo", N, MAX_HILOS, t_p)
            res["serial"][N], res["paralelo"][N] = resumen(t_s), resumen(t_p)

    # --- Tabla de tiempos ---
    print(f"\n  {'N':>6} | {'serial (s)':>16} | {'CPU ' + str(MAX_HILOS) + 'h (s)':>16} | "
        f"{'GPU kernel (ms)':>17} | {'H2D (ms)':>9} | {'D2H (ms)':>9} | {'transf./total':>13} | {'Ginter/s':>8}")
    for N in LISTA_N:
        k, h, d = res["kernel"][N], res["h2d"][N], res["d2h"][N]
        tot = k[0] + h[0] + d[0]
        txt_s = txt_p = f"{'-':>16}"
        if N in res["serial"]:
            s, p = res["serial"][N], res["paralelo"][N]
            txt_s = f"{s[0]:9.4f} ({s[1]:.4f})"
            txt_p = f"{p[0]:9.4f} ({p[1]:.4f})"
        print(f"  {N:6d} | {txt_s} | {txt_p} | {k[0] * 1e3:9.3f} ({k[1] * 1e3:.3f}) | "
              f"{h[0] * 1e3:9.3f} | {d[0] * 1e3:9.3f} | {(h[0] + d[0]) / tot:13.0%} | {N * N / k[0] / 1e9:8.1f}")

    # --- Tabla de speedups (cociente de medianas) ---
    print("\n  Speedup de la GPU = T_referencia / T_GPU   (\"kernel\": datos ya en la GPU; "
        "\"total\": kernel + H2D + D2H)")
    print(f"  {'N':>6} | {'kernel vs serial':>16} | {'kernel vs CPU ' + str(MAX_HILOS) + 'h':>17} | "
        f"{'total vs serial':>15} | {'total vs CPU ' + str(MAX_HILOS) + 'h':>16}")
    sp = {"k_ser": {}, "k_par": {}, "t_ser": {}, "t_par": {}}
    for N in LISTA_N:
        if N not in res["serial"]:
            continue
        tk = res["kernel"][N][0]
        tt = tk + res["h2d"][N][0] + res["d2h"][N][0]
        ts, tp = res["serial"][N][0], res["paralelo"][N][0]
        sp["k_ser"][N], sp["k_par"][N] = ts / tk, tp / tk
        sp["t_ser"][N], sp["t_par"][N] = ts / tt, tp / tt
        print(f"  {N:6d} | {sp['k_ser'][N]:16.1f} | {sp['k_par'][N]:17.1f} | "
            f"{sp['t_ser'][N]:15.1f} | {sp['t_par'][N]:16.1f}")
    print("  Nota: en la simulacion real posiciones y velocidades permanecen en la GPU; solo se copian "
        "al registrar (cada 'cada' pasos),\n  asi que el speedup \"kernel\" es el representativo y el "
        "\"total\" es el limite inferior si se copiara en cada paso.")
    print("  Nota: la eficiencia por hilo no aplica en GPU (rubrica); se reporta speedup frente a serial y "
        "frente a la CPU paralela.")

    # --- 5. Sensibilidad al tamano de bloque ---
    print(
        f"\n--- 5. Sensibilidad al tamaño de bloque (N = {N_TPB}, mediana de {REPS_GPU} rep.) ---")
    res_tpb = {}
    for tpb in LISTA_TPB:
        t = medir_gpu(lanzador(N_TPB, tpb), REPS_GPU)
        guardar("gpu_tpb", N_TPB, tpb, t)
        res_tpb[tpb] = resumen(t)
        m, iqr, lo, hi = res_tpb[tpb]
        print(f"  hilos por bloque = {tpb:4d}: {m * 1e3:8.3f} ms   IQR {iqr * 1e3:.3f}   "
              f"min-max {lo * 1e3:.3f}-{hi * 1e3:.3f}")

    # --- Chequeo de coherencia: misma config. (N_TPB, tpb0) medida en dos fases distintas ---
    if N_TPB in res["kernel"] and tpb0 in res_tpb:
        a, b = res["kernel"][N_TPB][0], res_tpb[tpb0][0]
        dif = abs(a / b - 1)
        if dif > TOL_AVISO:
            print(f"  AVISO: el kernel de N = {N_TPB} difiere {dif:.0%} entre el barrido principal "
                  f"({a * 1e3:.2f} ms) y la prueba de TPB ({b * 1e3:.2f} ms): posible throttling o ruido; "
                "repite con el equipo enchufado y enfriado.")
        else:
            print(
                f"  Coherencia OK: kernel de N = {N_TPB} difiere {dif:.0%} entre barrido principal y prueba de TPB.")

    ruta = os.path.join(RAIZ, "benchmarks", "resultados_gpu.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementacion", "N", "hilos_o_tpb", "rep", "tiempo_s"])
        w.writerows(filas)

    # --- Figura 2 x 2 ---
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    def serie(a, r, etiqueta, marcador):
        Ns = list(r)
        med = np.array([r[n][0] for n in Ns])
        lo = med - np.array([r[n][2] for n in Ns])
        hi = np.array([r[n][3] for n in Ns]) - med
        a.errorbar(Ns, med, yerr=[lo, hi],
                fmt=marcador + "-", capsize=3, label=etiqueta)

    a = ax[0, 0]
    serie(a, res["serial"], "CPU serial (Numba, 1 hilo)", "s")
    serie(a, res["paralelo"], f"CPU paralela ({MAX_HILOS} hilos)", "o")
    serie(a, res["kernel"], "GPU: solo kernel", "^")
    a.loglog(LISTA_N, [res["kernel"][N][0] + res["h2d"][N][0] + res["d2h"][N][0] for N in LISTA_N],
            "v--", label="GPU: kernel + transferencias")
    a.set_xlabel("Número de partículas (N)")
    a.set_ylabel("Tiempo por cálculo de fuerzas (s)")
    a.set_title("Tiempo vs. N (mediana; barras = min-max)")
    a.legend()
    a.grid(alpha=0.3, which="both")

    a = ax[0, 1]
    Ns_c = list(sp["k_ser"])
    a.loglog(Ns_c, [sp["k_ser"][n]
            for n in Ns_c], "^-", label="kernel vs CPU serial")
    a.loglog(Ns_c, [sp["k_par"][n] for n in Ns_c], "o-",
            label=f"kernel vs CPU {MAX_HILOS} hilos")
    a.loglog(Ns_c, [sp["t_ser"][n]
            for n in Ns_c], "v--", label="total vs CPU serial")
    a.loglog(Ns_c, [sp["t_par"][n] for n in Ns_c], "s--",
            label=f"total vs CPU {MAX_HILOS} hilos")
    a.axhline(1.0, color="k", ls=":", lw=0.8)
    a.set_xlabel("Número de partículas (N)")
    a.set_ylabel("Speedup de la GPU  (T_CPU / T_GPU)")
    a.set_title("Speedup de la GPU")
    a.legend()
    a.grid(alpha=0.3, which="both")

    a = ax[1, 0]
    tk = np.array([res["kernel"][N][0] for N in LISTA_N])
    th = np.array([res["h2d"][N][0] for N in LISTA_N])
    td = np.array([res["d2h"][N][0] for N in LISTA_N])
    tot = tk + th + td
    x = np.arange(len(LISTA_N))
    a.bar(x, th / tot * 100, label="copia host → GPU")
    a.bar(x, tk / tot * 100, bottom=th / tot * 100, label="kernel")
    a.bar(x, td / tot * 100, bottom=(th + tk) /
          tot * 100, label="copia GPU → host")
    a.set_xticks(x)
    a.set_xticklabels([str(n) for n in LISTA_N])
    a.set_xlabel("Número de partículas (N)")
    a.set_ylabel("% del tiempo total")
    a.set_title("Desglose: transferencias vs. cómputo")
    a.legend()

    a = ax[1, 1]
    med = np.array([res_tpb[t][0] for t in LISTA_TPB]) * 1e3
    lo = med - np.array([res_tpb[t][2] for t in LISTA_TPB]) * 1e3
    hi = np.array([res_tpb[t][3] for t in LISTA_TPB]) * 1e3 - med
    a.bar([str(t) for t in LISTA_TPB], med, yerr=[lo, hi], capsize=4)
    a.set_xlabel("Hilos por bloque (TPB)")
    a.set_ylabel("Tiempo del kernel (ms)")
    a.set_title(f"Sensibilidad al tamaño de bloque (N = {N_TPB})")

    fig.suptitle(f"Benchmark GPU: {dev}", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(RAIZ, "resultados", "benchmark_gpu.png"), dpi=130)
    print("\n  Guardado: resultados/benchmark_gpu.png y benchmarks/resultados_gpu.csv")


if __name__ == "__main__":
    if not cuda.is_available():
        sys.exit(
            "CUDA no esta disponible en este entorno (revisa que este activo nbody-gpu).")
    nombre, hw = info_hardware_gpu()
    print("\n".join(hw) + "\n")
    with open(os.path.join(RAIZ, "resultados", "hardware_gpu.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(hw) + "\n")
    prueba_correccion()
    prueba_energia_float32()
    print("\n  Guardado: benchmarks/verificacion_gpu.csv")
    if "--solo-verificacion" in sys.argv:
        sys.exit(0)     # no toca resultados_gpu.csv ni las figuras
    prueba_rendimiento(nombre)

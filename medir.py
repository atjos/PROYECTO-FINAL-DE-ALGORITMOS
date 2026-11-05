"""
medir.py - Script maestro: regenera TODOS los CSV, tablas y figuras del proyecto.

Uso:
    python medir.py --all              # CPU + GPU + tablas + figuras (todo, con video)
    python medir.py --all --rapido     # lo mismo con tamaños reducidos; las mediciones van a
                                       # benchmarks/rapido/ y resultados/rapido/ (no tocan los resultados publicados)
    python medir.py --cpu              # solo mediciones de CPU (y reanaliza las tablas)
    python medir.py --gpu              # solo mediciones de GPU (si hay CUDA; y reanaliza las tablas)
    python medir.py --analizar         # reanaliza los CSV existentes: speedup, Amdahl y tablas
    python medir.py --perfil           # perfilado del paso de simulacion (fuerzas vs energia vs resto)
    python medir.py --figuras          # figuras de fisica y colision (condiciones, validacion, video)
    python medir.py --figuras --sin-video   # igual, pero sin renderizar el video (mucho mas rapido)

Que genera cada paso (ver tambien la tabla "figura -> comando" del README):
    --cpu       benchmarks/resultados_cpu_*_c*.csv, benchmarks/verificacion_cpu.csv,
                resultados/benchmark_cpu.png, resultados/hardware.txt
    --gpu       benchmarks/resultados_gpu.csv, benchmarks/verificacion_gpu.csv,
                resultados/benchmark_gpu.png, resultados/hardware_gpu.txt
    --perfil    resultados/tabla_perfil.csv (tambien lo ejecutan --cpu y --all)
    --analizar  resultados/analisis_cpu.png, tabla_speedup_cpu.csv,
                tabla_baselines_cpu.csv, tabla_repeticiones.csv
    --figuras   resultados/condiciones_iniciales.png, estabilidad_galaxia.png,
                validacion_dt.png, colision_final.png, colision.mp4
"""

import sys
import os
import re
import argparse
import subprocess
from importlib import metadata

RAIZ = os.path.dirname(os.path.abspath(__file__))
PRUEBAS_DIR = os.path.join(RAIZ, "pruebas")


def _buscar_script(nombre):
    for sub in ["pruebas", "benchmarks", "."]:
        ruta = os.path.join(RAIZ, sub, nombre)
        if os.path.isfile(ruta):
            return ruta
    return None


def ejecutar_comando(comando, descripcion):
    print("\n" + "=" * 70)
    print(f" >> INICIANDO: {descripcion}")
    print(f" >> COMANDO:   {' '.join(comando)}")
    print("=" * 70 + "\n", flush=True)
    # PYTHONPATH con la raiz del proyecto: los scripts de pruebas/ hacen "from src import ..."
    entorno = dict(os.environ)
    entorno["PYTHONPATH"] = RAIZ + os.pathsep + entorno.get("PYTHONPATH", "")
    res = subprocess.run(comando, cwd=RAIZ, env=entorno)
    if res.returncode != 0:
        print(f"\n[ERROR] Fallo al ejecutar: {descripcion} (codigo {res.returncode})", file=sys.stderr)
        return False
    return True


def _script(nombre, args=(), descripcion=None):
    ruta = _buscar_script(nombre)
    if not ruta:
        print(f"[ERROR] No se encontro '{nombre}'.", file=sys.stderr)
        return False
    return ejecutar_comando([sys.executable, ruta, *args], descripcion or nombre)


# ---------------------------------------------------------------------------
# Comprobacion del entorno
# ---------------------------------------------------------------------------
def _version_instalada(paquete):
    for nombre in (paquete, paquete.replace("-", "_")):
        try:
            return metadata.version(nombre)
        except metadata.PackageNotFoundError:
            continue
    return None


def comprobar_entorno():
    """Compara las versiones instaladas con las fijadas en requirements.txt y AVISA de las diferencias.
    No detiene la medicion: el aviso queda en pantalla para que se corrija antes de reportar resultados
    (las versiones usadas en cada medicion tambien las guardan resultados/hardware*.txt)."""
    ruta = os.path.join(RAIZ, "requirements.txt")
    if not os.path.isfile(ruta):
        return True
    distintas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s#]+)", linea)
            if not m:
                continue
            paquete, pedida = m.groups()
            instalada = _version_instalada(paquete)
            if instalada != pedida:
                distintas.append((paquete, pedida, instalada or "no instalado"))
    if distintas:
        print("\n" + "!" * 70)
        print(" [AVISO] Las versiones instaladas no coinciden con requirements.txt:")
        for paquete, pedida, instalada in distintas:
            print(f"    {paquete:<14} requirements.txt: {pedida:<10} instalada: {instalada}")
        print(" Las mediciones quedarian hechas con otras versiones que las documentadas en el README.")
        print(" Activa el entorno indicado (conda activate nbody-gpu) o actualiza README y requirements.txt.")
        print("!" * 70 + "\n", flush=True)
        return False
    return True


# ---------------------------------------------------------------------------
# Mediciones
# ---------------------------------------------------------------------------
def medir_cpu(rapido=False, corridas=4, pausa=20):
    args = ["--rapido"] if rapido else ["--corridas", str(corridas), "--pausa", str(pausa)]
    return _script("benchmark_cpu.py", args, "Experimentos de CPU (Numba serial vs Numba paralelo)")


def perfilar(rapido=False):
    """Perfilado del paso de simulacion: que fraccion del tiempo son las fuerzas (criterio 3)."""
    return _script("perfil_serial.py", ["--rapido"] if rapido else [],
                   "Perfilado del paso de simulacion (resultados/tabla_perfil.csv)")


def medir_gpu(rapido=False):
    try:
        from numba import cuda
        if not cuda.is_available():
            print("[AVISO] CUDA no esta disponible en este entorno. Se omite el benchmark de GPU.", file=sys.stderr)
            return False
    except ImportError:
        print("[AVISO] Numba CUDA no esta instalado. Se omite el benchmark de GPU.", file=sys.stderr)
        return False
    return _script("benchmark_gpu.py", ["--rapido"] if rapido else [],
                   "Experimentos de GPU (kernel CUDA con tiling vs CPU)")


# ---------------------------------------------------------------------------
# Tablas y figuras derivadas
# ---------------------------------------------------------------------------
def analizar():
    """Todo lo que se calcula a partir de los CSV de benchmarks/."""
    pasos = [
        ("analizar_cpu.py", "Speedup, eficiencia y ajuste de Amdahl (analisis_cpu.png, tabla_speedup_cpu.csv)"),
        ("baselines_cpu.py", "Speedup frente a cada referencia (tabla_baselines_cpu.csv)"),
        ("resumen_repeticiones.py", "Repeticiones por configuracion (tabla_repeticiones.csv)"),
    ]
    return [_script(n, (), d) for n, d in pasos]


def figuras(rapido=False, con_video=True):
    """Figuras de fisica y de la colision (no dependen de los benchmarks)."""
    resultados = [
        _script("prueba_condiciones.py", (),
                "Condiciones iniciales y estabilidad de la galaxia (condiciones_iniciales.png, estabilidad_galaxia.png)"),
        _script("validacion_fisica.py", (),
                "Validacion fisica: orden 2, reversibilidad y energia (validacion_dt.png)"),
    ]
    if con_video:
        args = ["--rapido"] if rapido else []
        resultados.append(_script("visualizar.py", args,
                                  "Render de la colision (colision_final.png, colision.mp4)"))
    else:
        print("\n[INFO] Se omite el render del video (--sin-video): colision_final.png y colision.mp4 no se regeneran.")
    return resultados


def main():
    parser = argparse.ArgumentParser(
        description="Script maestro de medicion: regenera CSV, tablas y figuras.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Mediciones de CPU (y reanalisis de tablas)")
    parser.add_argument("--gpu", action="store_true", help="Mediciones de GPU (y reanalisis de tablas)")
    parser.add_argument("--all", action="store_true", help="CPU + GPU + tablas + figuras")
    parser.add_argument("--perfil", action="store_true", help="Perfilado del paso de simulacion (fuerzas vs resto)")
    parser.add_argument("--analizar", action="store_true", help="Reanalizar los CSV existentes (speedup, Amdahl, tablas)")
    parser.add_argument("--figuras", action="store_true", help="Figuras de fisica y de la colision")
    parser.add_argument("--sin-video", action="store_true", help="Con --figuras o --all: no renderiza el video")
    parser.add_argument("--rapido", action="store_true", help="Ejecucion reducida para verificar que todo corre")
    parser.add_argument("--corridas", type=int, default=4, help="Corridas de CPU (default: 4)")
    parser.add_argument("--pausa", type=float, default=20.0, help="Pausa entre corridas de CPU, en s (default: 20)")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nEjemplo sugerido: python medir.py --all --rapido")
        return 0

    resultados = {}          # paso -> True/False
    hubo_mediciones = False

    if args.all or args.cpu or args.gpu:
        comprobar_entorno()

    if args.all or args.cpu:
        resultados["mediciones CPU"] = medir_cpu(args.rapido, args.corridas, args.pausa)
        hubo_mediciones = True
    if args.all or args.gpu:
        resultados["mediciones GPU"] = medir_gpu(args.rapido)
        hubo_mediciones = True
    if args.all or args.cpu or args.perfil:
        resultados["perfilado del paso"] = perfilar(args.rapido)
    if args.all or args.analizar or hubo_mediciones:
        for nombre, ok in zip(("analisis de speedup/Amdahl", "tabla de referencias", "tabla de repeticiones"),
                              analizar()):
            resultados[nombre] = ok
    if args.all or args.figuras:
        for nombre, ok in zip(("condiciones iniciales", "validacion fisica", "render de la colision"),
                              figuras(args.rapido, not args.sin_video)):
            resultados[nombre] = ok

    print("\n" + "=" * 70)
    print(" >> RESUMEN")
    for paso, ok in resultados.items():
        print(f"    [{'OK' if ok else 'FALLO'}] {paso}")
    print(" >> Datos en 'benchmarks/'; tablas y figuras en 'resultados/'.")
    print("=" * 70 + "\n")
    return 0 if all(resultados.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

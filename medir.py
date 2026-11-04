"""
Uso:
    python medir.py --cpu          # Ejecuta los experimentos de CPU (3 corridas por defecto)
    python medir.py --gpu          # Ejecuta los experimentos de GPU (si CUDA está disponible)
    python medir.py --all          # Ejecuta CPU y GPU sucesivamente
    python medir.py --analizar     # Analiza los CSV generados y calcula Speedup y Amdahl
    python medir.py --rapido       # Modo de prueba rápida con tamaños reducidos
"""

import sys
import os
import argparse
import subprocess

RAIZ = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.join(RAIZ, "benchmarks")
PRUEBAS_DIR = os.path.join(RAIZ, "pruebas")

# Resolver rutas para scripts delegados (soporta si están en pruebas/ o benchmarks/)
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
    res = subprocess.run(comando, cwd=RAIZ)
    if res.returncode != 0:
        print(f"\n[ERROR] Ocurrió un fallo al ejecutar: {descripcion} (código {res.returncode})", file=sys.stderr)
        return False
    return True

def medir_cpu(rapido=False, corridas=3, pausa=20):
    script = _buscar_script("benchmark_cpu.py")
    if not script:
        sys.exit("Error: No se encontró 'benchmark_cpu.py'.")
    
    cmd = [sys.executable, script]
    if rapido:
        cmd.append("--rapido")
    else:
        cmd.extend(["--corridas", str(corridas), "--pausa", str(pausa)])
    
    return ejecutar_comando(cmd, "Experimentos de CPU (Numba Serial vs Numba Paralelo)")

def medir_gpu(rapido=False):
    # Verificar si CUDA está disponible antes de lanzar
    try:
        from numba import cuda
        if not cuda.is_available():
            print("[AVISO] CUDA no está disponible en este entorno. Omitiendo benchmark de GPU.", file=sys.stderr)
            return False
    except ImportError:
        print("[AVISO] Numba CUDA no está instalado. Omitiendo benchmark de GPU.", file=sys.stderr)
        return False

    script = _buscar_script("benchmark_gpu.py")
    if not script:
        sys.exit("Error: No se encontró 'benchmark_gpu.py'.")
    
    cmd = [sys.executable, script]
    if rapido:
        cmd.append("--rapido")
    
    return ejecutar_comando(cmd, "Experimentos de GPU (CUDA Tiling Kernel vs CPU)")

def analizar_cpu():
    script = _buscar_script("analizar_cpu.py")
    if not script:
        sys.exit("Error: No se encontró 'analizar_cpu.py'.")
    
    cmd = [sys.executable, script]
    return ejecutar_comando(cmd, "Análisis de Speedup, Eficiencia y Ajuste de Amdahl")

def main():
    parser = argparse.ArgumentParser(
        description="Script maestro de medición de desempeño y generación de CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cpu", action="store_true", help="Ejecutar mediciones de CPU")
    parser.add_argument("--gpu", action="store_true", help="Ejecutar mediciones de GPU")
    parser.add_argument("--all", action="store_true", help="Ejecutar todas las mediciones (CPU + GPU + Análisis)")
    parser.add_argument("--analizar", action="store_true", help="Ejecutar solo el análisis de resultados existentes")
    parser.add_argument("--rapido", action="store_true", help="Ejecución reducida para verificación rápida")
    parser.add_argument("--corridas", type=int, default=3, help="Número de corridas para CPU (default: 3)")
    parser.add_argument("--pausa", type=float, default=20.0, help="Pausa en segundos entre corridas CPU para enfriamiento (default: 20)")

    args = parser.parse_args()

    # Si se corre sin argumentos, mostrar ayuda o por defecto correr --all
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nEjemplo sugerido: python medir.py --all --rapido")
        return

    exito = True

    if args.all:
        exito_cpu = medir_cpu(rapido=args.rapido, corridas=args.corridas, pausa=args.pausa)
        if exito_cpu:
            analizar_cpu()
        medir_gpu(rapido=args.rapido)
    else:
        if args.cpu:
            exito_cpu = medir_cpu(rapido=args.rapido, corridas=args.corridas, pausa=args.pausa)
            if exito_cpu:
                analizar_cpu()
        if args.gpu:
            medir_gpu(rapido=args.rapido)
        if args.analizar:
            analizar_cpu()

    print("\n" + "=" * 70)
    print(" >> PROCESO DE MEDICIÓN FINALIZADO.")
    print(" >> Resultados guardados en 'benchmarks/' y 'resultados/'.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
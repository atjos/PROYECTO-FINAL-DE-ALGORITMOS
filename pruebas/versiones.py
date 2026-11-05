"""
versiones.py - Lista las versiones de TODAS las herramientas del proyecto y las guarda en
resultados/versiones.txt (para pegarlas en el README y en el informe).

Uso:  python pruebas/versiones.py
"""
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# paquetes de requirements.txt (obligatorios y opcionales) + dependencias que influyen en los resultados
PAQUETES = ["numpy", "numba", "llvmlite", "numba-cuda", "cuda-python", "cuda-bindings",
            "matplotlib", "scipy", "pillow", "opencv-python", "imageio", "psutil"]


def version_paquete(nombre):
    for n in (nombre, nombre.replace("-", "_")):
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return "no instalado"


def ejecutar(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip()
    except Exception:
        return None


def main():
    L = []
    L.append(f"Sistema operativo : {platform.platform()}")
    L.append(f"Procesador        : {platform.processor()}")
    L.append(f"Python            : {sys.version.split()[0]}  ({sys.executable})")
    L.append("")
    L.append("Paquetes de Python:")
    for p in PAQUETES:
        L.append(f"  {p:<14} {version_paquete(p)}")

    L.append("")
    L.append("GPU / CUDA:")
    try:
        from numba import cuda
        if cuda.is_available():
            dev = cuda.get_current_device()
            nombre = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
            L.append(f"  GPU                      : {nombre}")
            L.append(f"  Capacidad de computo     : {dev.compute_capability[0]}.{dev.compute_capability[1]}")
            L.append(f"  Multiprocesadores (SM)   : {dev.MULTIPROCESSOR_COUNT}")
            rt = cuda.runtime.get_version()
            L.append(f"  CUDA runtime (numba)     : {rt[0]}.{rt[1]}")
            drv = cuda.driver.get_version() if hasattr(cuda, "driver") and hasattr(cuda.driver, "get_version") else None
            if drv:
                L.append(f"  CUDA soportado por driver: {drv[0]}.{drv[1]}")
        else:
            L.append("  CUDA no disponible en este entorno")
    except Exception as e:
        L.append(f"  No se pudo consultar CUDA ({type(e).__name__}: {e})")
    smi = ejecutar(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    L.append(f"  Driver NVIDIA            : {smi if smi else 'nvidia-smi no disponible'}")

    L.append("")
    L.append("Programas externos:")
    for prog, arg in (("ffmpeg", "-version"), ("conda", "--version")):
        ruta = shutil.which(prog)
        if ruta:
            salida = ejecutar([ruta, arg])
            L.append(f"  {prog:<8} {salida.splitlines()[0] if salida else '?'}")
        else:
            L.append(f"  {prog:<8} no encontrado en el PATH")

    texto = "\n".join(L)
    print(texto)
    os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
    ruta = os.path.join(RAIZ, "resultados", "versiones.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(f"\nGuardado: {ruta}")


if __name__ == "__main__":
    main()

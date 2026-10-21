from numba import cuda
import numpy as np
import numba
from numba import njit, prange
import time

print("NumPy:", np.__version__)
print("Numba:", numba.__version__)


@njit(parallel=True)
def suma_cuadrados(n):
    total = 0.0
    for i in prange(n):
        total += i * i
    return total


suma_cuadrados(10)  # la primera llamada compila
t0 = time.time()
r = suma_cuadrados(100_000_000)
print("Resultado:", r, "| Tiempo:", round(time.time() - t0, 3), "s")
print("Hilos disponibles:", numba.config.NUMBA_NUM_THREADS)

print("CUDA disponible:", cuda.is_available())

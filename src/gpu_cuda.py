"""
gpu_cuda.py - Version GPU (CUDA con Numba) del simulador de N cuerpos.

Ideas clave
-----------
* UN HILO DE GPU POR PARTICULA: el hilo i calcula a_i (como el prange de la version CPU,
  pero con miles de hilos en vez de 12).
* MEMORIA COMPARTIDA (tiling): los hilos de un bloque cargan juntos un "tile" de
  tpb particulas en la memoria compartida (rapida) y todos lo reutilizan, en vez de
  leer cada particula de la memoria global tpb veces.
* FLOAT32: la GTX 1650 hace FP64 ~32 veces mas lento que FP32 (lo mostro cuda.detect()).
* TODO EL ESTADO VIVE EN LA GPU: las posiciones y velocidades no viajan a la CPU en cada
  paso, solo cuando se guarda una "foto" (cada 'cada' pasos).

Convencion: pos4 es (N, 4) float32 con columnas [x, y, z, masa]; vel y acc son (N, 3) float32.
CUIDADO con los literales: en Numba un 0.0 suelto es float64 y contaminaria el calculo con
operaciones FP64 lentas, por eso se usan float32(0.0), float32(1.0), etc.
"""
import math
import functools
import numpy as np
from numba import cuda, float32

G = 1.0
TPB_POR_DEFECTO = 128


@functools.lru_cache(maxsize=None)
def construir_kernels(tpb):
    """Compila los kernels para un tamano de bloque 'tpb' (la memoria compartida exige que sea constante)."""

    @cuda.jit(fastmath=True)
    def k_acc(pos4, acc, n, eps2):
        sh = cuda.shared.array((tpb, 4), dtype=float32)     # tile en memoria compartida
        tx = cuda.threadIdx.x
        i = cuda.grid(1)
        cero = float32(0.0)
        uno = float32(1.0)

        xi = cero
        yi = cero
        zi = cero
        if i < n:
            xi = pos4[i, 0]
            yi = pos4[i, 1]
            zi = pos4[i, 2]
        ax = cero
        ay = cero
        az = cero

        n_tiles = (n + tpb - 1) // tpb
        for tile in range(n_tiles):
            j = tile * tpb + tx
            if j < n:                                        # cada hilo carga UNA particula del tile
                sh[tx, 0] = pos4[j, 0]
                sh[tx, 1] = pos4[j, 1]
                sh[tx, 2] = pos4[j, 2]
                sh[tx, 3] = pos4[j, 3]
            else:                                            # relleno: masa 0 => no aporta fuerza
                sh[tx, 0] = cero
                sh[tx, 1] = cero
                sh[tx, 2] = cero
                sh[tx, 3] = cero
            cuda.syncthreads()                               # esperar a que el tile este completo

            for k in range(tpb):                             # interaccion con las tpb particulas del tile
                dx = sh[k, 0] - xi
                dy = sh[k, 1] - yi
                dz = sh[k, 2] - zi
                r2 = dx * dx + dy * dy + dz * dz + eps2
                inv_r = uno / math.sqrt(r2)
                s = sh[k, 3] * inv_r * inv_r * inv_r
                ax += dx * s                                 # si j == i: dx=dy=dz=0 -> aporta 0
                ay += dy * s
                az += dz * s
            cuda.syncthreads()                               # no sobreescribir el tile antes de que todos terminen

        if i < n:
            acc[i, 0] = ax
            acc[i, 1] = ay
            acc[i, 2] = az

    @cuda.jit(fastmath=True)
    def k_pot(pos4, u, n, eps2):
        """u[i] = m_i * sum_{j != i} m_j / sqrt(r_ij^2 + eps^2)   (la energia potencial es -0.5*G*sum u)."""
        sh = cuda.shared.array((tpb, 4), dtype=float32)
        tx = cuda.threadIdx.x
        i = cuda.grid(1)
        cero = float32(0.0)
        uno = float32(1.0)
        xi = cero
        yi = cero
        zi = cero
        mi = cero
        if i < n:
            xi = pos4[i, 0]
            yi = pos4[i, 1]
            zi = pos4[i, 2]
            mi = pos4[i, 3]
        suma = cero

        n_tiles = (n + tpb - 1) // tpb
        for tile in range(n_tiles):
            j = tile * tpb + tx
            if j < n:
                sh[tx, 0] = pos4[j, 0]
                sh[tx, 1] = pos4[j, 1]
                sh[tx, 2] = pos4[j, 2]
                sh[tx, 3] = pos4[j, 3]
            else:
                sh[tx, 0] = cero
                sh[tx, 1] = cero
                sh[tx, 2] = cero
                sh[tx, 3] = cero
            cuda.syncthreads()
            for k in range(tpb):
                if tile * tpb + k != i:                      # excluir la autointeraccion
                    dx = sh[k, 0] - xi
                    dy = sh[k, 1] - yi
                    dz = sh[k, 2] - zi
                    r2 = dx * dx + dy * dy + dz * dz + eps2
                    suma += sh[k, 3] * (uno / math.sqrt(r2))
            cuda.syncthreads()
        if i < n:
            u[i] = mi * suma

    @cuda.jit
    def k_kick(vel, acc, n, h):
        i = cuda.grid(1)
        if i < n:
            vel[i, 0] += acc[i, 0] * h
            vel[i, 1] += acc[i, 1] * h
            vel[i, 2] += acc[i, 2] * h

    @cuda.jit
    def k_drift(pos4, vel, n, dt):
        i = cuda.grid(1)
        if i < n:
            pos4[i, 0] += vel[i, 0] * dt
            pos4[i, 1] += vel[i, 1] * dt
            pos4[i, 2] += vel[i, 2] * dt

    return k_acc, k_pot, k_kick, k_drift


def empaquetar(pos, masas):
    """(N,3) y (N,) float64  ->  (N,4) float32 [x, y, z, m]."""
    pos4 = np.empty((len(masas), 4), dtype=np.float32)
    pos4[:, :3] = pos
    pos4[:, 3] = masas
    return pos4


def aceleraciones(pos, masas, eps, tpb=TPB_POR_DEFECTO):
    """Aceleraciones en la GPU, devueltas como float64 en la CPU (sirve para validar contra NumPy)."""
    k_acc = construir_kernels(tpb)[0]
    n = len(masas)
    d_pos4 = cuda.to_device(empaquetar(pos, masas))
    d_acc = cuda.device_array((n, 3), dtype=np.float32)
    k_acc[(n + tpb - 1) // tpb, tpb](d_pos4, d_acc, n, np.float32(eps * eps))
    return d_acc.copy_to_host().astype(np.float64)


def simular(pos, vel, masas, eps, dt, n_pasos, cada=10, guardar_pos=True, tpb=TPB_POR_DEFECTO):
    """Igual que fisica.simular / cpu_paralelo.simular, pero todo el calculo ocurre en la GPU."""
    k_acc, k_pot, k_kick, k_drift = construir_kernels(tpb)
    n = len(masas)
    bloques = (n + tpb - 1) // tpb
    eps2 = np.float32(eps * eps)
    dt32 = np.float32(dt)
    medio = np.float32(0.5 * dt)

    d_pos4 = cuda.to_device(empaquetar(pos, masas))
    d_vel = cuda.to_device(np.ascontiguousarray(vel, dtype=np.float32))
    d_acc = cuda.device_array((n, 3), dtype=np.float32)
    d_u = cuda.device_array(n, dtype=np.float32)

    k_acc[bloques, tpb](d_pos4, d_acc, n, eps2)

    tiempos, energias, P, L, snapshots = [], [], [], [], []

    def registrar(t):
        p4 = d_pos4.copy_to_host()
        v = d_vel.copy_to_host().astype(np.float64)
        k_pot[bloques, tpb](d_pos4, d_u, n, eps2)
        U = -0.5 * G * np.sum(d_u.copy_to_host().astype(np.float64))
        m = p4[:, 3].astype(np.float64)
        p = p4[:, :3].astype(np.float64)
        K = 0.5 * np.sum(m * np.sum(v**2, axis=1))
        tiempos.append(t)
        energias.append(K + U)
        P.append(np.sum(m[:, None] * v, axis=0))
        L.append(np.sum(m[:, None] * np.cross(p, v), axis=0))
        if guardar_pos:
            snapshots.append(p)

    registrar(0.0)
    for paso in range(1, n_pasos + 1):
        k_kick[bloques, tpb](d_vel, d_acc, n, medio)
        k_drift[bloques, tpb](d_pos4, d_vel, n, dt32)
        k_acc[bloques, tpb](d_pos4, d_acc, n, eps2)
        k_kick[bloques, tpb](d_vel, d_acc, n, medio)
        if paso % cada == 0:
            registrar(paso * dt)

    cuda.synchronize()
    p4 = d_pos4.copy_to_host()
    return {
        "t": np.array(tiempos), "E": np.array(energias),
        "P": np.array(P), "L": np.array(L),
        "pos": np.array(snapshots) if guardar_pos else None,
        "pos_final": p4[:, :3].astype(np.float64),
        "vel_final": d_vel.copy_to_host().astype(np.float64),
    }
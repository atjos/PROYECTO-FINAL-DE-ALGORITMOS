"""
cpu_paralelo.py - Version paralela en CPU (Numba) del simulador de N cuerpos.

Idea: cada aceleracion a_i se calcula de forma INDEPENDIENTE de las demas, asi que el
bucle exterior (sobre i) se reparte entre los hilos del procesador con prange.
Cada hilo escribe solo en su propia fila acc[i], por lo que no hay condiciones de carrera.

Se ofrecen dos compilaciones del mismo codigo, para poder comparar:
    aceleraciones_numba_serial   -> compilada, 1 solo hilo
    aceleraciones_numba          -> compilada, todos los hilos
"""
import numpy as np
from numba import njit, prange

G = 1.0


def _aceleraciones(pos, masas, eps):
    n = pos.shape[0]
    eps2 = eps * eps
    acc = np.empty((n, 3))
    for i in prange(n):                    # <- aqui se reparte el trabajo entre hilos
        xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            if j != i:
                dx = pos[j, 0] - xi
                dy = pos[j, 1] - yi
                dz = pos[j, 2] - zi
                r2 = dx * dx + dy * dy + dz * dz + eps2
                inv_r = 1.0 / np.sqrt(r2)
                s = masas[j] * inv_r * inv_r * inv_r
                ax += dx * s
                ay += dy * s
                az += dz * s
        acc[i, 0] = G * ax
        acc[i, 1] = G * ay
        acc[i, 2] = G * az
    return acc


def _energia_potencial(pos, masas, eps):
    n = pos.shape[0]
    eps2 = eps * eps
    U = 0.0
    for i in prange(n):
        suma = 0.0
        for j in range(n):
            if j != i:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                suma += masas[j] / np.sqrt(dx * dx + dy * dy + dz * dz + eps2)
        U += masas[i] * suma
    return -0.5 * G * U                    # 0.5: cada par se cuenta dos veces


# Mismo codigo, dos compilaciones. Con parallel=False, prange se comporta como range.
aceleraciones_numba_serial = njit(fastmath=True)(_aceleraciones)
aceleraciones_numba = njit(parallel=True, fastmath=True)(_aceleraciones)
energia_potencial_numba = njit(parallel=True, fastmath=True)(_energia_potencial)


def energia(pos, vel, masas, eps):
    """(E_total, K, U) sin crear matrices N x N: sirve para N grande."""
    K = 0.5 * np.sum(masas * np.sum(vel**2, axis=1))
    U = energia_potencial_numba(pos, masas, eps)
    return K + U, K, U


def paso_leapfrog(pos, vel, acc, masas, eps, dt, f_acc=aceleraciones_numba):
    vel = vel + 0.5 * dt * acc
    pos = pos + dt * vel
    acc = f_acc(pos, masas, eps)
    vel = vel + 0.5 * dt * acc
    return pos, vel, acc


def simular(pos, vel, masas, eps, dt, n_pasos, cada=10, guardar_pos=True,
            f_acc=aceleraciones_numba):
    """Igual que fisica.simular, pero con las fuerzas calculadas en paralelo."""
    pos = np.ascontiguousarray(pos, dtype=np.float64).copy()
    vel = np.ascontiguousarray(vel, dtype=np.float64).copy()
    masas = np.ascontiguousarray(masas, dtype=np.float64)
    acc = f_acc(pos, masas, eps)

    tiempos, energias, P, L, snapshots = [], [], [], [], []

    def registrar(t):
        tiempos.append(t)
        energias.append(energia(pos, vel, masas, eps)[0])
        P.append(np.sum(masas[:, None] * vel, axis=0))
        L.append(np.sum(masas[:, None] * np.cross(pos, vel), axis=0))
        if guardar_pos:
            snapshots.append(pos.copy())

    registrar(0.0)
    for n in range(1, n_pasos + 1):
        pos, vel, acc = paso_leapfrog(pos, vel, acc, masas, eps, dt, f_acc)
        if n % cada == 0:
            registrar(n * dt)

    return {
        "t": np.array(tiempos), "E": np.array(energias),
        "P": np.array(P), "L": np.array(L),
        "pos": np.array(snapshots) if guardar_pos else None,
        "pos_final": pos, "vel_final": vel,
    }
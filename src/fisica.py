import numpy as np

G = 1.0


def aceleraciones(pos, masas, eps):
    """a_i = G * sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)"""
    # diff[i, j] = r_j - r_i   -> forma (N, N, 3)
    diff = pos[None, :, :] - pos[:, None, :]
    r2 = np.sum(diff**2, axis=2) + eps**2
    inv_r3 = r2 ** (-1.5)
    np.fill_diagonal(inv_r3, 0.0)          # una partícula no se atrae a si misma
    peso = masas[None, :] * inv_r3         # (N, N): m_j / r^3
    return G * np.einsum("ij,ijk->ik", peso, diff)


def energia(pos, vel, masas, eps):
    """Devuelve (E_total, K, U) con el mismo softening que la fuerza."""
    K = 0.5 * np.sum(masas * np.sum(vel**2, axis=1))
    diff = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt(np.sum(diff**2, axis=2) + eps**2)
    inv_r = 1.0 / r
    np.fill_diagonal(inv_r, 0.0)
    # 0.5 porque la suma doble cuenta cada par dos veces
    U = -0.5 * G * np.sum(masas[:, None] * masas[None, :] * inv_r)
    return K + U, K, U


def momento_lineal(vel, masas):
    return np.sum(masas[:, None] * vel, axis=0)


def momento_angular(pos, vel, masas):
    return np.sum(masas[:, None] * np.cross(pos, vel), axis=0)


def paso_leapfrog(pos, vel, acc, masas, eps, dt):
    """Un paso kick-drift-kick. 'acc' es la aceleracion en las posiciones actuales."""
    vel = vel + 0.5 * dt * acc                 # kick (medio paso)
    pos = pos + dt * vel                       # drift (paso completo)
    acc = aceleraciones(pos, masas, eps)       # fuerzas en las nuevas posiciones
    vel = vel + 0.5 * dt * acc                 # kick (medio paso)
    return pos, vel, acc


def simular(pos, vel, masas, eps, dt, n_pasos, cada=10):
    """
    Integra n_pasos. Cada 'cada' pasos guarda: tiempo, energia, momento lineal
    y angular, y una copia de las posiciones.
    """
    pos = pos.copy()
    vel = vel.copy()
    acc = aceleraciones(pos, masas, eps)

    tiempos, energias, P, L, snapshots = [], [], [], [], []

    def registrar(t):
        tiempos.append(t)
        energias.append(energia(pos, vel, masas, eps)[0])
        P.append(momento_lineal(vel, masas))
        L.append(momento_angular(pos, vel, masas))
        snapshots.append(pos.copy())

    registrar(0.0)
    for n in range(1, n_pasos + 1):
        pos, vel, acc = paso_leapfrog(pos, vel, acc, masas, eps, dt)
        if n % cada == 0:
            registrar(n * dt)

    return {
        "t": np.array(tiempos),
        "E": np.array(energias),
        "P": np.array(P),
        "L": np.array(L),
        "pos": np.array(snapshots),
        "pos_final": pos,
        "vel_final": vel,
    }
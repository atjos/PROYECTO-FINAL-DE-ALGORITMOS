import numpy as np

G = 1.0


def matriz_rotacion(inclinacion, angulo_z):
    """Rota primero 'inclinacion' alrededor del eje x y luego 'angulo_z' alrededor del eje z (radianes)."""
    ci, si = np.cos(inclinacion), np.sin(inclinacion)
    cz, sz = np.cos(angulo_z), np.sin(angulo_z)
    Rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Rx


def crear_galaxia(n_estrellas, masa_nucleo=0.8, masa_disco=0.2, escala_disco=1.0,
                r_min=0.2, r_max=5.0, altura=0.05, dispersion=0.03,
                eps=0.05, sentido=+1, rng=None):
    """
    Galaxia de disco centrada en el origen, con el disco en el plano x-y.

    n_estrellas : numero de estrellas (el nucleo es 1 particula adicional)
    masa_nucleo, masa_disco : masas totales (en unidades de M0)
    escala_disco : longitud de escala R_d del disco exponencial
    r_min, r_max : radios minimo y maximo de las estrellas (en unidades de L0)
    altura       : desviacion estandar del espesor vertical (en unidades de R_d)
    dispersion   : dispersion de velocidades como fraccion de la velocidad circular
    eps          : softening (el mismo que usara la simulacion)
    sentido      : +1 gira antihorario, -1 horario

    Devuelve pos, vel, masas (el indice 0 es el nucleo).
    """
    rng = np.random.default_rng() if rng is None else rng
    Rd = escala_disco

    # --- 1. Radios: disco exponencial, Sigma ~ exp(-R/Rd)  <=>  R ~ Gamma(2, Rd) ---
    R = np.empty(0)
    while R.size < n_estrellas:
        candidatos = rng.gamma(shape=2.0, scale=Rd, size=2 * n_estrellas)
        candidatos = candidatos[(candidatos > r_min) & (candidatos < r_max)]
        R = np.concatenate([R, candidatos])
    R = R[:n_estrellas]
    R.sort()                                   # ordenados de adentro hacia afuera
    phi = rng.uniform(0, 2 * np.pi, n_estrellas)
    z = rng.normal(0.0, altura * Rd, n_estrellas)

    # --- 2. Posiciones ---
    x, y = R * np.cos(phi), R * np.sin(phi)

    # --- 3. Velocidad circular: v_c^2 = G M(<R) R^2 / (R^2 + eps^2)^(3/2) ---
    m_estrella = masa_disco / n_estrellas
    M_dentro = masa_nucleo + m_estrella * \
        np.arange(1, n_estrellas + 1)   # nucleo + estrellas mas internas
    vc = np.sqrt(G * M_dentro * R**2 / (R**2 + eps**2) ** 1.5)

    # direccion tangencial (antihoraria para sentido=+1) + pequena dispersion
    vx = -sentido * vc * np.sin(phi)
    vy = sentido * vc * np.cos(phi)
    vz = np.zeros(n_estrellas)
    sigma = dispersion * vc
    vx = vx + rng.normal(0, 1, n_estrellas) * sigma
    vy = vy + rng.normal(0, 1, n_estrellas) * sigma
    vz = vz + rng.normal(0, 1, n_estrellas) * sigma

    # --- 4. Ensamblar (nucleo en el indice 0) ---
    pos = np.vstack([[0.0, 0.0, 0.0], np.column_stack([x, y, z])])
    vel = np.vstack([[0.0, 0.0, 0.0], np.column_stack([vx, vy, vz])])
    masas = np.concatenate([[masa_nucleo], np.full(n_estrellas, m_estrella)])

    # --- 5. Llevar al sistema del centro de masa (el nucleo absorbe el pequeno desbalance) ---
    pos -= np.average(pos, axis=0, weights=masas)
    vel -= np.average(vel, axis=0, weights=masas)
    return pos, vel, masas


def crear_colision(n_por_galaxia=2000, separacion=12.0, parametro_impacto=3.0,
                frac_parabolica=0.6, razon_masa=1.0,
                incl1=0.0, incl2=np.radians(60), angulo1=0.0, angulo2=0.0,
                   sentido1=+1, sentido2=+1, eps=0.05, semilla=None, **kwargs):
    """
    Dos galaxias que se acercan en el plano x-y, en el sistema del centro de masa.

    separacion        : distancia inicial entre nucleos
    parametro_impacto : desviacion perpendicular (0 = frontal; grande = rasante)
    frac_parabolica   : velocidad relativa inicial como fraccion de la parabolica
                        sqrt(2 G M_total / separacion). < 1 => orbita ligada (terminan fusionandose)
    razon_masa        : masa de la galaxia 2 / masa de la galaxia 1
    incl*, angulo*    : orientacion de cada disco (radianes)
    sentido*          : +1 / -1 sentido de giro de cada disco
    kwargs            : parametros extra para crear_galaxia (masa_nucleo, escala_disco, ...)

    Devuelve pos, vel, masas, galaxia (0/1 por particula), es_nucleo (bool por particula).
    """
    rng = np.random.default_rng(semilla)
    n_est2 = n_por_galaxia

    p1, v1, m1 = crear_galaxia(
        n_por_galaxia, eps=eps, sentido=sentido1, rng=rng, **kwargs)
    p2, v2, m2 = crear_galaxia(
        n_est2, eps=eps, sentido=sentido2, rng=rng, **kwargs)
    m2 = m2 * razon_masa                       # galaxia 2 mas (o menos) masiva

    R1, R2 = matriz_rotacion(incl1, angulo1), matriz_rotacion(incl2, angulo2)
    p1, v1 = p1 @ R1.T, v1 @ R1.T
    p2, v2 = p2 @ R2.T, v2 @ R2.T

    M1, M2 = m1.sum(), m2.sum()
    Mtot = M1 + M2
    v_rel = frac_parabolica * np.sqrt(2.0 * G * Mtot / separacion)

    # posicion y velocidad relativas (galaxia 2 respecto a la 1); el movimiento es en -x
    r_rel = np.array([separacion, parametro_impacto, 0.0])
    u_rel = np.array([-v_rel, 0.0, 0.0])
    # sistema del centro de masa
    p1 = p1 + (-M2 / Mtot) * r_rel
    p2 = p2 + (M1 / Mtot) * r_rel
    v1 = v1 + (-M2 / Mtot) * u_rel
    v2 = v2 + (M1 / Mtot) * u_rel

    pos = np.vstack([p1, p2])
    vel = np.vstack([v1, v2])
    masas = np.concatenate([m1, m2])
    galaxia = np.concatenate([np.zeros(len(m1), int), np.ones(len(m2), int)])
    es_nucleo = np.zeros(len(masas), bool)
    es_nucleo[0] = True
    es_nucleo[len(m1)] = True
    return pos, vel, masas, galaxia, es_nucleo

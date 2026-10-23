from src.condiciones import crear_colision
import numpy as np
import sys
import os
import time
import argparse
import shutil
import subprocess
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from scipy.ndimage import gaussian_filter, uniform_filter1d
except ImportError:
    gaussian_filter = uniform_filter1d = None

EPS = 0.05


def elegir_backend(pedido):
    if pedido in ("auto", "gpu"):
        try:
            from numba import cuda
            if cuda.is_available():
                return "gpu"
        except Exception:
            pass
        if pedido == "gpu":
            sys.exit(
                "Pediste --backend gpu pero CUDA no esta disponible (entorno nbody-gpu?).")
    if pedido in ("auto", "cpu"):
        try:
            import numba  # noqa: F401
            return "cpu"
        except Exception:
            if pedido == "cpu":
                sys.exit("Pediste --backend cpu pero Numba no esta instalado.")
    return "numpy"


def trayectoria(pos, vel, masas, dt, n_pasos, cada, backend):
    """Devuelve P (F,N,3) y V (F,N,3) float32, con F = n_pasos//cada + 1 fotos."""
    n = len(masas)
    F = n_pasos // cada + 1
    P = np.empty((F, n, 3), np.float32)
    V = np.empty((F, n, 3), np.float32)
    t0 = time.perf_counter()

    def aviso(paso):
        if paso % max(cada * 20, 1) == 0 or paso == n_pasos:
            seg = time.perf_counter() - t0
            print(
                f"  paso {paso}/{n_pasos}  ({100 * paso / n_pasos:3.0f} %)  {seg:6.1f} s", flush=True)

    if backend == "gpu":
        import gpu_cuda as gpu
        from numba import cuda
        tpb = gpu.TPB_POR_DEFECTO
        k_acc, _, k_kick, k_drift = gpu.construir_kernels(tpb)
        bloques = (n + tpb - 1) // tpb
        eps2, dt32, medio = np.float32(
            EPS * EPS), np.float32(dt), np.float32(0.5 * dt)
        d_pos4 = cuda.to_device(gpu.empaquetar(pos, masas))
        d_vel = cuda.to_device(np.ascontiguousarray(vel, dtype=np.float32))
        d_acc = cuda.device_array((n, 3), dtype=np.float32)
        k_acc[bloques, tpb](d_pos4, d_acc, n, eps2)
        P[0], V[0] = d_pos4.copy_to_host()[:, :3], d_vel.copy_to_host()
        for paso in range(1, n_pasos + 1):
            k_kick[bloques, tpb](d_vel, d_acc, n, medio)
            k_drift[bloques, tpb](d_pos4, d_vel, n, dt32)
            k_acc[bloques, tpb](d_pos4, d_acc, n, eps2)
            k_kick[bloques, tpb](d_vel, d_acc, n, medio)
            if paso % cada == 0:
                P[paso // cada], V[paso // cada] = d_pos4.copy_to_host()[:,
                                                                        :3], d_vel.copy_to_host()
                aviso(paso)
    else:
        if backend == "cpu":
            import cpu_paralelo as mod
            f_acc = mod.aceleraciones_numba

            def step(p, v, a): return mod.paso_leapfrog(
                p, v, a, masas, EPS, dt, f_acc)
        else:
            import fisica as mod
            f_acc = mod.aceleraciones
            def step(p, v, a): return mod.paso_leapfrog(
                p, v, a, masas, EPS, dt)
        p = np.ascontiguousarray(pos, dtype=np.float64).copy()
        v = np.ascontiguousarray(vel, dtype=np.float64).copy()
        masas = np.ascontiguousarray(masas, dtype=np.float64)
        a = f_acc(p, masas, EPS)
        P[0], V[0] = p, v
        for paso in range(1, n_pasos + 1):
            p, v, a = step(p, v, a)
            if paso % cada == 0:
                P[paso // cada], V[paso // cada] = p, v
                aviso(paso)
    return P, V


def desenfocar(img, sigma):
    if sigma <= 0:
        return img
    if cv2 is not None:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    return gaussian_filter(img, (sigma, sigma, 0))


def paleta(puntos, colores):
    """Interpola una lista de colores hex -> funcion t en [0,1] -> RGB (M,3) en [0,1]."""
    c = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
                for h in colores])
    xs = np.linspace(0, 1, len(c))
    return lambda t: np.stack([np.interp(t, xs, c[:, k]) for k in range(3)], -1)


PAL_VEL = paleta(None, ["#20407d", "#3a75a4", "#8ec1dc",
                "#f2e2a8", "#e6883c", "#b8321e"])
PAL_DEN = paleta(None, ["#0d131f", "#1e334a", "#4a779d", "#c49a45", "#fff3d6"])
COL_GAL = np.array([[0.28, 0.65, 0.95], [0.95, 0.52, 0.18]])


class Render:
    def __init__(self, P, V, galaxia, nucleo, dt_foto, W, H, modo, estela, giro, brillo):
        self.P, self.V, self.gal, self.nuc = P, V, galaxia, nucleo
        self.F = len(P)
        self.W, self.H, self.modo = W, H, modo
        self.estela, self.giro, self.brillo = estela, giro, brillo
        self.dt_foto = dt_foto
        self.s = W / 1280.0
        self.idx_nuc = np.where(nucleo)[0]
        self.estrellas = ~nucleo
        rapidez = np.linalg.norm(
            V[::max(1, self.F // 20)][:, self.estrellas], axis=2)
        self.vref = np.percentile(rapidez, 97)
        anchos = np.empty(self.F)
        for f in range(self.F):
            r = np.linalg.norm(P[f][self.estrellas], axis=1)
            rn = np.linalg.norm(P[f][self.idx_nuc], axis=1).max()
            anchos[f] = max(np.percentile(r, 97), 1.3 * rn, 4.0) * \
                1.1 * (W / H)
        k = max(3, self.F // 12)
        self.anchos = uniform_filter1d(
            anchos, k, mode="nearest") if uniform_filter1d else anchos
        self.ref_exp = None

    def angulos(self, f):
        u = f / max(self.F - 1, 1)
        az = np.radians(-25 + self.giro * u)
        el = np.radians(28 + 22 * np.sin(2 * np.pi * u))
        return az, el

    def proyectar(self, pos, az, el):
        ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
        q = pos @ (Rx @ Rz).T
        return q[:, 0], q[:, 1], q[:, 2]

    def pixeles(self, x, y, ancho):
        ix = np.rint((x / ancho * 0.5 + 0.5) * self.W).astype(np.int64)
        iy = np.rint(self.H / 2 - y / ancho * self.W / 2).astype(np.int64)
        return ix, iy

    def splat(self, ix, iy, col, w):
        W, H = self.W, self.H
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        idx = iy[ok] * W + ix[ok]
        canales = [np.bincount(idx, weights=col[ok, c]
                               * w[ok], minlength=W * H) for c in range(3)]
        return np.stack(canales, -1).reshape(H, W, 3).astype(np.float32)

    def colores(self, f, ix, iy, prof, ancho):
        vel = self.V[f]
        if self.modo == "galaxia":
            col = COL_GAL[self.gal]
            rap = np.clip(np.linalg.norm(vel, axis=1) /
                        self.vref, 0, 1.3)[:, None]
            col = col * (0.75 + 0.35 * rap)
        elif self.modo == "densidad":
            cnt = self.splat(ix, iy, np.ones(
                (len(ix), 3), np.float32), np.ones(len(ix), np.float32))[..., 0]
            dens = desenfocar(cnt[..., None], 5 * self.s)[..., 0]
            ok = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
            d = np.zeros(len(ix))
            d[ok] = dens[iy[ok], ix[ok]]
            t = np.log1p(
                d) / np.log1p(np.percentile(d[ok], 99.5) + 1e-9) if ok.any() else d
            col = PAL_DEN(np.clip(t, 0, 1))
        else:
            col = PAL_VEL(np.clip(np.linalg.norm(
                vel, axis=1) / self.vref, 0, 1))
        fz = np.clip(1.0 + 0.25 * prof / ancho, 0.6, 1.4)[:, None]
        return col * fz

    def frame(self, f):
        W, H, s = self.W, self.H, self.s
        az, el = self.angulos(f)
        ancho = self.anchos[f]
        est = self.estrellas
        x, y, z = self.proyectar(self.P[f], az, el)
        ix, iy = self.pixeles(x, y, ancho)
        col = self.colores(f, ix, iy, z, ancho)[est]
        base_w = self.brillo * np.ones(est.sum(), np.float32)

        img = self.splat(ix[est], iy[est], col, base_w)

        for k in range(1, self.estela + 1):
            if f - k < 0:
                break
            xk, yk, _ = self.proyectar(self.P[f - k], az, el)
            ik, jk = self.pixeles(xk, yk, ancho)
            peso = base_w * 0.55 * np.exp(-k / max(self.estela / 2.5, 1e-6))
            img += self.splat(ik[est], jk[est], col, peso)

        img = img + 0.6 * desenfocar(img, 1.5 * s) + \
            1.4 * desenfocar(img, 6.0 * s)

        lum = img.mean(-1)
        ref = np.percentile(lum, 99.6) + 1e-9
        self.ref_exp = ref if self.ref_exp is None else 0.9 * self.ref_exp + 0.1 * ref
        out = 1.0 - np.exp(-2.4 * img / self.ref_exp)
        out = out ** 0.8

        trazo = np.zeros((H, W, 3), np.float32)
        paso = max(1, f // 250)
        for g, gi in enumerate(self.idx_nuc):
            ks = np.arange(f, -1, -paso)
            xs, ys, _ = self.proyectar(self.P[ks][:, gi], az, el)
            a, b = self.pixeles(xs, ys, ancho)
            edad = (f - ks) / max(f, 1)
            w = (1.0 - edad) ** 1.5 * 0.9
            c = np.tile(COL_GAL[g] * 0.6 + 0.4, (len(ks), 1))
            trazo += self.splat(a, b, c, w.astype(np.float32))
        trazo = np.clip(trazo + 0.6 * desenfocar(trazo, 1.5 * s), 0, 1) * 0.8
        out = 1.0 - (1.0 - out) * (1.0 - trazo)

        nx, ny = ix[self.idx_nuc], iy[self.idx_nuc]
        n_col = np.ones((len(nx), 3), np.float32)
        capa = self.splat(nx, ny, n_col, np.ones(len(nx), np.float32))
        s1, s2 = 2.0 * s, 7.5 * s
        glow = desenfocar(capa, s1) * (1.5 * np.pi * s1 ** 2) + \
            0.25 * desenfocar(capa, s2) * (1.5 * np.pi * s2 ** 2)
        tinte = np.array([1.0, 0.97, 0.9], np.float32)
        out = 1.0 - (1.0 - out) * (1.0 - np.clip(glow, 0, 1) * tinte)

        yy, xx = np.mgrid[0:H, 0:W]
        vin = 1.0 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 +
                            ((yy - H / 2) / (H / 2)) ** 2) / 2
        out = np.clip(out * vin[..., None] + 0.008, 0, 1)
        return (out * 255).astype(np.uint8)

    def con_texto(self, img, f):
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            try:
                fuente = ImageFont.load_default(size=max(12, int(16 * self.s)))
            except TypeError:
                fuente = ImageFont.load_default()
            t = f * self.dt_foto
            d.text((14 * self.s, 10 * self.s),
                f"t = {t:5.1f}", fill=(190, 200, 215), font=fuente)
            modo = {"velocidad": "Mapeo: velocidad orbital", "galaxia": "Mapeo: galaxia de origen",
                    "densidad": "Mapeo: densidad superficial"}[self.modo]
            d.text((14 * self.s, 10 * self.s + 22 * self.s), f"N = {len(self.P[0])}   {modo}",
                fill=(120, 130, 150), font=fuente)
            return np.asarray(im)
        except ImportError:
            return img


class Escritor:
    def __init__(self, ruta, W, H, fps):
        self.W, self.H, self.fps = W, H, fps
        self.proc = None
        self.gif = None
        if shutil.which("ffmpeg"):
            self.ruta = ruta + ".mp4"
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{W}x{H}", "-r", str(
                    fps), "-i", "-", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "18", self.ruta], stdin=subprocess.PIPE)
        else:
            import imageio
            self.ruta = ruta + ".gif"
            self.gif = imageio.get_writer(
                self.ruta, mode="I", duration=1.0 / fps, loop=0)
            print("  (ffmpeg no encontrado: se escribe un GIF; instala ffmpeg para MP4)")

    def escribir(self, img):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.gif.append_data(img)

    def cerrar(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.gif.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000,
                    help="estrellas por galaxia (default 4000)")
    ap.add_argument("--pasos", type=int, default=6000,
                    help="pasos de integracion (default 6000 -> t = 30)")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--cada", type=int, default=5,
                    help="pasos entre cuadros (default 5)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument(
        "--color", choices=["velocidad", "galaxia", "densidad"], default="velocidad")
    ap.add_argument("--estela", type=int, default=5)
    ap.add_argument("--giro", type=float, default=120.0)
    ap.add_argument("--brillo", type=float, default=1.0)
    ap.add_argument(
        "--backend", choices=["auto", "gpu", "cpu", "numpy"], default="auto")
    ap.add_argument("--salida", default=os.path.join(RAIZ,
                    "resultados", "colision"))
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--vivo", action="store_true",
                    help="ventana interactiva en vez de video")
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()

    if a.rapido:
        a.n, a.pasos, a.cada, a.res = min(
            a.n, 400), min(a.pasos, 1500), 10, "640x360"
    W, H = (int(v) // 2 * 2 for v in a.res.lower().split("x"))
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)

    pos, vel, masas, gal, nuc = crear_colision(
        n_por_galaxia=a.n, semilla=1, eps=EPS)
    ruta_cache = os.path.join(RAIZ, "resultados", "trayectoria.npz")
    clave = np.array([a.n, a.pasos, a.cada, a.dt])
    if a.cache and os.path.exists(ruta_cache) and np.allclose(np.load(ruta_cache)["clave"], clave):
        print("Usando trayectoria en cache:", ruta_cache)
        d = np.load(ruta_cache)
        P, V = d["P"], d["V"]
    else:
        backend = elegir_backend(a.backend)
        print(
            f"Simulando N = {len(masas)} cuerpos, {a.pasos} pasos, backend = {backend}")
        P, V = trayectoria(pos, vel, masas, a.dt, a.pasos, a.cada, backend)
        if a.cache:
            np.savez(ruta_cache, P=P, V=V, clave=clave)

    r = Render(P, V, gal, nuc, a.dt * a.cada, W, H,
            a.color, a.estela, a.giro, a.brillo)
    F = len(P)

    png = a.salida + "_final.png"
    f_fin = int(F * 0.85)
    img = r.con_texto(r.frame(f_fin), f_fin)
    if cv2 is not None:
        cv2.imwrite(png, img[..., ::-1])
    else:
        from PIL import Image
        Image.fromarray(img).save(png)
    print("Guardado:", png)
    r.ref_exp = None

    if a.vivo:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        fig = plt.figure(figsize=(W / 100, H / 100), facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        im = ax.imshow(r.con_texto(r.frame(0), 0))

        def act(f):
            im.set_data(r.con_texto(r.frame(f), f))
            return (im,)
        _ = FuncAnimation(fig, act, frames=F, interval=1000 /
                        a.fps, blit=True, repeat=True)
        plt.show()
        return

    esc = Escritor(a.salida, W, H, a.fps)
    t0 = time.perf_counter()
    for f in range(F):
        esc.escribir(r.con_texto(r.frame(f), f))
        if f % 25 == 0 or f == F - 1:
            print(
                f"  render {f + 1}/{F}  ({time.perf_counter() - t0:5.0f} s)", flush=True)
    esc.cerrar()
    print("Guardado:", esc.ruta,
        f"({F} cuadros, {F / a.fps:.1f} s a {a.fps} fps)")


if __name__ == "__main__":
    main()

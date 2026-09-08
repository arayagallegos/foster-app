"""tests/synthetic_cloud.py — Nube sintética del 'Observatorio' con verdad conocida.

Soporta casos difíciles opcionales (desnivel de suelo, junta cúpula-tambor realista,
contrafuertes, interior) para evaluar la precisión de la segmentación con métricas.
"""
from __future__ import annotations

import numpy as np

R_TAMBOR = 2.5
H_MURO = 3.0
R_SUELO = 6.0


def _ring(r: float, z_min: float, z_max: float, n: int, rng) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.uniform(z_min, z_max, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def _dome(r: float, cz: float, n: int, rng) -> np.ndarray:
    """Semiesfera (muestreo uniforme en área) centrada en (0, 0, cz)."""
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    cos_phi = rng.uniform(0.0, 1.0, n)
    sin_phi = np.sqrt(1.0 - cos_phi**2)
    return np.column_stack([
        r * sin_phi * np.cos(theta),
        r * sin_phi * np.sin(theta),
        cz + r * cos_phi,
    ])


def _dome_realista(r: float, cz: float, n: int, rng) -> np.ndarray:
    """
    Cúpula con un leve sobresalto (overhang) y una FALDA que baja por debajo de
    su ecuador (z < cz). Reproduce el mecanismo real observado en la nube: la
    banda de transición de la cúpula queda bajo el plano z = sz y el corte
    horizontal rígido del orquestador la misclasifica como tambor (a cualquier
    tolerancia). La verdad de referencia de estos puntos sigue siendo cúpula.
    """
    r_over = r + 0.02                              # leve overhang, dentro de la banda
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    cos_phi = rng.uniform(-0.15, 1.0, n)           # phi hasta ~99°: falda bajo el ecuador
    sin_phi = np.sqrt(1.0 - cos_phi**2)
    return np.column_stack([
        r_over * sin_phi * np.cos(theta),
        r_over * sin_phi * np.sin(theta),
        cz + r_over * cos_phi,                     # cos_phi<0 -> z < cz (bajo el ecuador)
    ])


def _disk(r_max: float, z: float, n: int, rng) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = r_max * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), np.full(n, z)])


def _contrafuertes(n_buttress: int, n_pts: int, rng) -> np.ndarray:
    """Prismas verticales pegados a la cara exterior del muro, repartidos en ángulo."""
    out = []
    for k in range(n_buttress):
        ang = 2.0 * np.pi * k / max(n_buttress, 1)
        base = np.array([R_TAMBOR * np.cos(ang), R_TAMBOR * np.sin(ang), 0.0])
        du = np.array([np.cos(ang), np.sin(ang), 0.0])           # radial
        dv = np.array([-np.sin(ang), np.cos(ang), 0.0])          # tangencial
        u = rng.uniform(0.0, 0.4, n_pts)                         # 40 cm hacia afuera
        v = rng.uniform(-0.25, 0.25, n_pts)                      # 50 cm de ancho
        z = rng.uniform(0.0, H_MURO, n_pts)
        pts = base + np.outer(u, du) + np.outer(v, dv)
        pts[:, 2] = z
        out.append(pts)
    return np.vstack(out) if out else np.empty((0, 3))


def make_synthetic_foster(
    seed: int = 0,
    noise: float = 0.01,
    outlier_frac: float = 0.2,
    r_int: float | None = None,
    pendiente_suelo_deg: float = 0.0,
    junta_realista: bool = False,
    n_contrafuertes: int = 0,
    n_interior: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    dome_fn = _dome_realista if junta_realista else _dome
    partes = [
        (_disk(R_SUELO, 0.0, 8000, rng), 1),
        (_ring(R_TAMBOR, 0.0, H_MURO, 8000, rng), 2),
        (dome_fn(R_TAMBOR, H_MURO, 6000, rng), 3),
    ]
    if r_int is not None:
        partes.append((_ring(r_int, 0.0, H_MURO, 4000, rng), 2))
    if n_contrafuertes > 0:
        partes.append((_contrafuertes(n_contrafuertes, 800, rng), 0))
    if n_interior > 0:
        interior = np.column_stack([
            rng.uniform(-R_TAMBOR * 0.7, R_TAMBOR * 0.7, n_interior),
            rng.uniform(-R_TAMBOR * 0.7, R_TAMBOR * 0.7, n_interior),
            rng.uniform(0.2, H_MURO * 0.8, n_interior),
        ])
        partes.append((interior, 0))

    pts = np.vstack([p for p, _ in partes])
    labels = np.concatenate([np.full(len(p), lab) for p, lab in partes])
    pts = pts + rng.normal(0.0, noise, pts.shape)

    # Desnivel: inclinar todo el modelo alrededor del eje Y por la pendiente del suelo
    if pendiente_suelo_deg != 0.0:
        a = np.radians(pendiente_suelo_deg)
        # z += tan(a) * x  (plano de suelo deja de ser horizontal)
        pts = pts.copy()
        pts[:, 2] = pts[:, 2] + np.tan(a) * pts[:, 0]

    if outlier_frac > 0.0:
        n_out = int(outlier_frac * len(pts))
        outliers = np.column_stack([
            rng.uniform(-6.0, 6.0, n_out),
            rng.uniform(-6.0, 6.0, n_out),
            rng.uniform(0.0, 7.0, n_out),
        ])
        pts = np.vstack([pts, outliers])
        labels = np.concatenate([labels, np.zeros(n_out, dtype=int)])

    orden = rng.permutation(len(pts))
    return pts[orden], labels[orden]


def make_synthetic_foster_dificil(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Nube con todos los casos difíciles activados (escenario de evaluación)."""
    return make_synthetic_foster(
        seed=seed,
        pendiente_suelo_deg=4.0,
        junta_realista=True,
        n_contrafuertes=6,
        n_interior=1500,
        outlier_frac=0.15,
    )


# ------------------------------------------------------------------ #
# Nube fiel a escala real, con 6 clases (para segment_by_profile)      #
# ------------------------------------------------------------------ #
# Dimensiones medidas del Foster:
R_T = 4.41          # radio exterior del tambor
ESP_MURO = 0.21     # espesor de muro medido
R_COR = 4.70        # radio de la cornisa (sobresale)
H_MUR = 1.5         # altura del muro
H_COR = 0.9         # altura de la cornisa
ESP_CUP = 0.05      # espesor de la cáscara de la cúpula


def make_synthetic_foster_6clases(
    seed: int = 0, noise: float = 0.01, outlier_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nube a escala real con verdad de 6 clases y sin coincidencias de clase:
    0=interior, 1=suelo, 2=tambor, 3=cupula, 4=cornisa, 5=contrafuerte.
    """
    rng = np.random.default_rng(seed)
    partes = [
        (_disk(9.0, 0.0, 9000, rng), 1),                        # suelo
        (_ring(R_T, 0.0, H_MUR, 7000, rng), 2),                 # tambor ext
        (_ring(R_T - ESP_MURO, 0.0, H_MUR, 4000, rng), 2),      # tambor int
        (_ring(R_COR, H_MUR, H_MUR + H_COR, 3000, rng), 4),     # cornisa (radio máx)
        (_dome(R_T, H_MUR + H_COR, 6000, rng), 3),              # cúpula ext
        (_dome(R_T - ESP_CUP, H_MUR + H_COR, 4000, rng), 3),    # cúpula int
    ]
    # contrafuertes: prismas radiales que SOBRESALEN del tambor
    for k in range(6):
        ang = 2 * np.pi * k / 6
        du = np.array([np.cos(ang), np.sin(ang), 0.0])
        dv = np.array([-np.sin(ang), np.cos(ang), 0.0])
        base = R_T * du
        u = rng.uniform(0.03, 0.5, 700)           # arranca 3 cm fuera del muro (sin coincidir)
        v = rng.uniform(-0.2, 0.2, 700)
        zz = rng.uniform(0.0, H_MUR - 0.10, 700)  # termina bajo la cornisa (sin coincidir)
        cf = base + np.outer(u, du) + np.outer(v, dv)
        cf[:, 2] = zz
        partes.append((cf, 5))
    # interior: puntos sueltos dentro del tambor
    interior = np.column_stack([
        rng.uniform(-R_T * 0.6, R_T * 0.6, 1500),
        rng.uniform(-R_T * 0.6, R_T * 0.6, 1500),
        rng.uniform(0.3, H_MUR + H_COR, 1500),
    ])
    partes.append((interior, 0))

    pts = np.vstack([p for p, _ in partes])
    labels = np.concatenate([np.full(len(p), lab) for p, lab in partes])
    pts = pts + rng.normal(0.0, noise, pts.shape)
    if outlier_frac > 0:
        n = int(outlier_frac * len(pts))
        out = np.column_stack([rng.uniform(-9, 9, n), rng.uniform(-9, 9, n),
                               rng.uniform(0, 8, n)])
        pts = np.vstack([pts, out])
        labels = np.concatenate([labels, np.zeros(n, dtype=int)])
    orden = rng.permutation(len(pts))
    return pts[orden], labels[orden]


# ------------------------------------------------------------------ #
# Interior sintético con verdad conocida (para refine_interior/DBSCAN) #
# ------------------------------------------------------------------ #
def make_synthetic_interior(
    seed: int = 0, noise: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], float]:
    """Interior sintético a escala real, con verdad conocida.
    Devuelve (pts, truth, centro_cupula, r_cupula).
      7 = compuertas (parche denso sobre la esfera -> superficie plana),
      3 = vigas       (costillas finas que huggean la esfera -> lineales, adosadas),
      6 = descartado  (telescopio central + bloques de equipo, despegados del domo).
    """
    rng = np.random.default_rng(seed)
    centro = np.array([0.0, 0.0, 0.0])
    r = 6.0
    partes: list[tuple[np.ndarray, int]] = []

    # Compuertas: parche denso sobre la esfera (ventana angular acotada) -> planar
    n = 6000
    th = rng.uniform(0.2, 1.1, n)
    ph = rng.uniform(0.2, 0.9, n)
    sinp = np.sin(ph)
    panel = np.column_stack([r * sinp * np.cos(th), r * sinp * np.sin(th), r * np.cos(ph)])
    partes.append((panel, 7))

    # Vigas/costillas: arcos finos sobre la esfera, lejos del cenit para no tocar el panel
    for a in (3.5, 4.3):
        m = 1500
        ph2 = rng.uniform(0.5, 1.3, m)
        rib = np.column_stack([
            r * np.sin(ph2) * np.cos(a), r * np.sin(ph2) * np.sin(a), r * np.cos(ph2)])
        rib += rng.normal(0.0, 0.01, rib.shape)   # barra fina => cluster lineal
        partes.append((rib, 3))

    # Telescopio: cilindro vertical central (no toca la cáscara)
    m = 4000
    tz = rng.uniform(-3.0, 1.0, m)
    ta = rng.uniform(0.0, 2 * np.pi, m)
    tr = 0.5 * np.sqrt(rng.uniform(0.0, 1.0, m))
    tele = np.column_stack([tr * np.cos(ta), tr * np.sin(ta), tz])
    partes.append((tele, 6))

    # Equipo: dos bloques cerca del piso (no tocan la cáscara)
    for cx, cy in ((2.0, 0.0), (-1.5, 1.5)):
        m = 1200
        blk = np.column_stack([
            rng.uniform(cx - 0.4, cx + 0.4, m),
            rng.uniform(cy - 0.4, cy + 0.4, m),
            rng.uniform(-3.2, -2.6, m)])
        partes.append((blk, 6))

    pts = np.vstack([p for p, _ in partes])
    truth = np.concatenate([np.full(len(p), t) for p, t in partes])
    pts = pts + rng.normal(0.0, noise, pts.shape)
    orden = rng.permutation(len(pts))
    return pts[orden], truth[orden], tuple(centro), r

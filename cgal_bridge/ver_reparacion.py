"""
ver_reparacion.py — Muestra visualmente lo que hizo CGAL: malla con hoyo (izq)
vs malla reparada (der). Tambien escribe ambas a output/spikes/ como .ply.

Uso:  python cgal_bridge/ver_reparacion.py
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

import cgal_bridge  # noqa: E402
from test_cgal_bridge import esfera_con_hoyo  # noqa: E402


def a_pv(V, F):
    """(V,F) -> PolyData de PyVista (caras con el prefijo de 3 vertices)."""
    caras = np.hstack([np.full((len(F), 1), 3, dtype=np.int64),
                       F.astype(np.int64)]).ravel()
    return pv.PolyData(V, caras)


def main():
    V, F = esfera_con_hoyo()
    antes = cgal_bridge.mesh_stats(V, F)
    V2, F2 = cgal_bridge.fill_holes(V, F, fair=True)
    despues = cgal_bridge.mesh_stats(V2, F2)

    m1, m2 = a_pv(V, F), a_pv(V2, F2)

    out = AQUI.parent / "output/spikes"
    out.mkdir(parents=True, exist_ok=True)
    m1.save(str(out / "malla_con_hoyo.ply"))
    m2.save(str(out / "malla_reparada.ply"))
    print(f"-> {out}\\malla_con_hoyo.ply  y  malla_reparada.ply")

    pl = pv.Plotter(shape=(1, 2), window_size=(1400, 700))
    pl.subplot(0, 0)
    pl.add_text(f"ANTES  |  hoyos={antes['n_holes']}  "
                f"cerrada={antes['is_closed']}", font_size=11)
    pl.add_mesh(m1, color="lightgray", show_edges=True, edge_color="gray")
    pl.subplot(0, 1)
    pl.add_text(f"DESPUES (CGAL)  |  hoyos={despues['n_holes']}  "
                f"cerrada={despues['is_closed']}", font_size=11)
    pl.add_mesh(m2, color="lightblue", show_edges=True, edge_color="gray")
    pl.link_views()
    pl.camera_position = "xz"
    print("Gira la malla: el parche nuevo es la zona con triangulos mas finos.")
    pl.show()


if __name__ == "__main__":
    main()

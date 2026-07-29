"""
ver_hoyos.py — Pinta CADA ciclo de borde (hoyo) de un color distinto.

Sirve para responder empiricamente: los huecos que se ven dispersos por la
superficie, ¿son hoyos independientes, o partes de un mismo contorno que
serpentea por la malla? Si comparten color, son el mismo hoyo.

Uso:
    python cgal_bridge/ver_hoyos.py output/mallas/compuertas_ULTIMA.ply
"""
import sys
from collections import defaultdict

import numpy as np
import pyvista as pv


def ciclos_de_borde(m):
    """Agrupa las aristas de frontera en ciclos conectados (componentes)."""
    bordes = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                     manifold_edges=False, non_manifold_edges=False)
    if bordes.n_cells == 0:
        return bordes, np.array([]), 0
    # etiqueta cada arista con su componente conexa
    conn = bordes.connectivity()
    etq = np.asarray(conn.cell_data["RegionId"])
    return bordes, etq, int(etq.max()) + 1


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else "output/mallas/compuertas_ULTIMA.ply"
    m = pv.read(ruta)
    bordes, etq, n = ciclos_de_borde(m)
    print(f"{ruta}: {m.n_points:,} V, {m.n_cells:,} F")
    print(f"aristas de borde: {bordes.n_cells:,}  ->  {n} ciclos independientes")

    cuenta = defaultdict(int)
    for e in etq:
        cuenta[int(e)] += 1
    top = sorted(cuenta.items(), key=lambda t: -t[1])
    print(f"\n  {'ciclo':>6} {'aristas':>9} {'% del total':>12}")
    for cid, c in top[:10]:
        print(f"  {cid:>6} {c:>9,} {100*c/len(etq):>11.1f}%")

    bordes.cell_data["ciclo"] = etq.astype(float)
    pl = pv.Plotter()
    pl.add_mesh(m, color="lightgray", smooth_shading=True)
    pl.add_mesh(bordes, scalars="ciclo", cmap="tab20", line_width=6,
                show_scalar_bar=False)
    pl.add_text(f"{n} ciclos de borde ({bordes.n_cells:,} aristas)\n"
                f"mismo color = MISMO hoyo", font_size=11)
    print("\nMismo color = mismo hoyo. Gira para inspeccionar.")
    pl.show()


if __name__ == "__main__":
    main()

"""
ver_malla.py — Visor de diagnostico de mallas: superficie + hoyos marcados.

Renderiza la malla como SUPERFICIE (no como nube) y resalta en ROJO los bordes
de hoyo (aristas de frontera). Asi se ve DONDE estan los problemas.

Uso:
    python cgal_bridge/ver_malla.py output/mallas/compuertas_reparada.ply
    python cgal_bridge/ver_malla.py output/mallas/compuertas_cruda.ply output/mallas/compuertas_reparada.ply
"""
import sys
from pathlib import Path

import pyvista as pv


def cargar(ruta):
    m = pv.read(str(ruta))
    if not isinstance(m, pv.PolyData) or m.n_cells == 0:
        raise SystemExit(f"{ruta}: no tiene caras (se leyo como nube de puntos)")
    return m


def panel(pl, m, titulo):
    # bordes de hoyo = aristas de frontera (pertenecen a una sola cara)
    bordes = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                     manifold_edges=False, non_manifold_edges=False)
    nm = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                 manifold_edges=False, non_manifold_edges=True)
    pl.add_mesh(m, color="lightgray", smooth_shading=True, show_edges=False)
    if bordes.n_cells:
        pl.add_mesh(bordes, color="red", line_width=4)
    if nm.n_cells:
        pl.add_mesh(nm, color="magenta", line_width=4)
    pl.add_text(f"{titulo}\n{m.n_points:,} V  {m.n_cells:,} F\n"
                f"ROJO = bordes de hoyo ({bordes.n_cells:,} aristas)\n"
                f"MAGENTA = no-manifold ({nm.n_cells:,} aristas)",
                font_size=9)
    print(f"{titulo}: {m.n_points:,} V, {m.n_cells:,} F | "
          f"aristas de frontera={bordes.n_cells:,} | no-manifold={nm.n_cells:,}")


def main():
    rutas = sys.argv[1:]
    if not rutas:
        rutas = ["output/mallas/compuertas_reparada.ply"]
    mallas = [(Path(r).stem, cargar(r)) for r in rutas]

    pl = pv.Plotter(shape=(1, len(mallas)), window_size=(700 * len(mallas), 750))
    for i, (nom, m) in enumerate(mallas):
        pl.subplot(0, i)
        panel(pl, m, nom)
    if len(mallas) > 1:
        pl.link_views()
    print("\nGira la malla. Rojo = agujeros; magenta = aristas no-manifold.")
    pl.show()


if __name__ == "__main__":
    main()

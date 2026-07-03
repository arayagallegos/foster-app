"""
freecad_toy_geometry.py — Geometría de prueba: cilindro hueco + cúpula semiesférica.

Se ejecuta con el intérprete de FreeCAD (NO con el venv):
    FreeCADCmd.exe freecad_toy_geometry.py <salida.step>

FreeCAD trabaja en mm: las dimensiones en metros se multiplican por M=1000.
"""
import sys

import Part  # noqa: disponible solo dentro de FreeCAD
from FreeCAD import Vector

M = 1000.0  # metros → milímetros
RADIO = 2.5 * M
ALTURA_MURO = 3.0 * M
ESPESOR_MURO = 0.30 * M
ESPESOR_CUPULA = 0.15 * M


def build() -> "Part.Shape":
    # Muro: tubo cilíndrico (cilindro exterior menos interior)
    ext = Part.makeCylinder(RADIO, ALTURA_MURO)
    interior = Part.makeCylinder(RADIO - ESPESOR_MURO, ALTURA_MURO)
    muro = ext.cut(interior)

    # Cúpula: casquete semiesférico hueco apoyado sobre el muro
    # makeSphere(radio, centro, eje, lat_min, lat_max): 0..90 = hemisferio superior
    centro = Vector(0, 0, ALTURA_MURO)
    dome_ext = Part.makeSphere(RADIO, centro, Vector(0, 0, 1), 0, 90)
    dome_int = Part.makeSphere(RADIO - ESPESOR_CUPULA, centro, Vector(0, 0, 1), 0, 90)
    cupula = dome_ext.cut(dome_int)

    solido = muro.fuse(cupula).removeSplitter()
    return solido


def main() -> None:
    out_path = sys.argv[-1]
    if not out_path.lower().endswith(".step"):
        raise SystemExit(f"Uso: FreeCADCmd freecad_toy_geometry.py <salida.step> "
                         f"(recibí: {out_path})")
    shape = build()
    if not shape.isValid():
        raise SystemExit("La geometría resultante no es un sólido válido.")
    shape.exportStep(out_path)
    print(f"[freecad] STEP exportado: {out_path}")


main()

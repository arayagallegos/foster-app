"""
test_cgal_bridge.py — Verifica que el puente CGAL compilo y funciona.

Criterio de exito del spike: tomar una malla CON un hoyo, rellenarlo con CGAL
desde Python, y confirmar que queda CERRADA (watertight).

Uso (venv activo, tras compilar segun COMPILACION.md):
    python cgal_bridge/test_cgal_bridge.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cgal_bridge
except ImportError as e:
    print("[X] No se pudo importar cgal_bridge.")
    print(f"    {e}")
    print("    Compila primero siguiendo cgal_bridge/COMPILACION.md")
    sys.exit(1)


def esfera_con_hoyo(n_lat=24, n_lon=48, quitar=6):
    """Malla de esfera UV a la que se le sacan caras del polo -> deja un hoyo."""
    lat = np.linspace(0, np.pi, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    V = [[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]
         for a in lat for b in lon]
    F = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            # se omiten las primeras filas de caras -> hoyo en el polo norte
            if i < quitar:
                continue
            F.append([a, b, c])
            F.append([b, d, c])
    return np.array(V, dtype=np.float64), np.array(F, dtype=np.int32)


def main():
    V, F = esfera_con_hoyo()
    print(f"Malla de prueba: {len(V)} vertices, {len(F)} caras\n")

    antes = cgal_bridge.mesh_stats(V, F)
    print("ANTES de reparar:")
    for k, v in antes.items():
        print(f"  {k:16s}: {v}")

    assert antes["n_holes"] > 0, "la malla de prueba deberia tener un hoyo"
    assert not antes["is_closed"], "no deberia estar cerrada aun"

    V2, F2 = cgal_bridge.fill_holes(V, F, fair=True)
    despues = cgal_bridge.mesh_stats(V2, F2)
    print(f"\nDESPUES de fill_holes:  ({len(V2)} vertices, {len(F2)} caras)")
    for k, v in despues.items():
        print(f"  {k:16s}: {v}")

    ok = despues["n_holes"] == 0 and despues["is_closed"]
    print("\n" + ("[OK] CGAL relleno el hoyo y la malla quedo CERRADA."
                  if ok else
                  "[X] El hoyo NO se cerro correctamente."))
    if despues["self_intersects"]:
        print("[!] Aviso: la malla resultante tiene auto-intersecciones.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

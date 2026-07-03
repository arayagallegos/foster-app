"""
fem_toy_test.py — Corre el pipeline FEM completo sobre la geometría de prueba.

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/fem_toy_test.py                      # malla 0.4 m
    python scripts/fem_toy_test.py --element-size 0.2   # estudio de convergencia
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.modules.fem import (  # noqa: E402
    MAMPOSTERIA_GENERICA,
    extract_base_nodes,
    generate_toy_step,
    mesh_step,
    parse_frequencies,
    run_ccx,
    write_modal_deck,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline FEM de prueba (toy)")
    ap.add_argument("--element-size", type=float, default=0.4, help="tamaño de elemento [m]")
    ap.add_argument("--n-modes", type=int, default=10)
    args = ap.parse_args()

    out = Path("output/toy")
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Geometria (FreeCADCmd -> .step)...")
    step = generate_toy_step(out / "toy.step")

    print(f"[2/4] Malla (gmsh, tamaño {args.element_size} m)...")
    info = mesh_step(step, out / "toy.inp", element_size=args.element_size)
    print(f"      {info.n_nodes:,} nodos, {info.n_elements:,} elementos C3D10")

    print("[3/4] Deck modal (CalculiX)...")
    base = extract_base_nodes(out / "toy.inp")
    print(f"      {len(base)} nodos empotrados en la base")
    deck = write_modal_deck(out / "toy.inp", out / "modal.inp", base,
                            MAMPOSTERIA_GENERICA, n_modes=args.n_modes)

    print("[4/4] Resolviendo (ccx)...")
    dat = run_ccx(deck)
    freqs = parse_frequencies(dat)

    print("\nFrecuencias naturales (base empotrada, mampostería genérica):")
    for i, f in enumerate(freqs, 1):
        print(f"  Modo {i:2d}: {f:8.3f} Hz")


if __name__ == "__main__":
    main()

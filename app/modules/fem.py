"""
fem.py — Módulo 5: Pipeline Scan-to-FEM (geometría → malla → análisis modal).

Etapas desacopladas por archivos:
  FreeCADCmd (geometría .step) → gmsh (malla .inp) → deck CalculiX → ccx → .dat

FreeCAD usa su propio intérprete Python, por eso la geometría se genera con un
script externo (scripts/freecad_toy_geometry.py) ejecutado como subprocess.
Todo lo demás corre en el venv del proyecto. Unidades SI: m, kg, s, Pa.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------------------------------ #
# Materiales (elástico lineal isotrópico, unidades SI)                 #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class Material:
    """Material elástico lineal isotrópico en unidades SI (Pa, kg/m3)."""
    name: str
    E: float
    nu: float
    rho: float


# Supuesto documentado en la tesis mientras el laboratorio confirma el material real.
MAMPOSTERIA_GENERICA = Material(name="MAMPOSTERIA", E=2.0e9, nu=0.2, rho=1800.0)


# ------------------------------------------------------------------ #
# Localización de herramientas externas                                #
# ------------------------------------------------------------------ #

def _find_tool(env_var: str, exe_name: str, install_hint: str) -> Path:
    env_val = os.environ.get(env_var)
    if env_val:
        p = Path(env_val)
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"{env_var}={env_val} no existe. Corrige la variable o elimínala."
        )
    for hit in sorted(glob.glob(rf"C:\Program Files\FreeCAD*\bin\{exe_name}"),
                      reverse=True):
        return Path(hit)
    raise FileNotFoundError(
        f"No se encontró {exe_name}. {install_hint} "
        f"O define la variable de entorno {env_var} con la ruta completa."
    )


def find_freecadcmd() -> Path:
    return _find_tool("FOSTER_FREECADCMD", "FreeCADCmd.exe",
                      "Instala FreeCAD 1.0 (freecad.org).")


def find_ccx() -> Path:
    return _find_tool("FOSTER_CCX", "ccx.exe",
                      "Viene en el bin de FreeCAD 1.0; si no, calculix.de.")


# ------------------------------------------------------------------ #
# Malla → deck de análisis modal                                       #
# ------------------------------------------------------------------ #

def extract_base_nodes(inp_path: Path, tol: float = 1e-6) -> list[int]:
    """
    Ids de los nodos de la base (z <= tol) leyendo el bloque *NODE del .inp.
    Sirve para construir el NSET del empotramiento sin depender de cómo
    gmsh exporta los grupos físicos.
    """
    base: list[int] = []
    in_nodes = False
    found_block = False
    for line in Path(inp_path).read_text().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("*NODE"):
            in_nodes = True
            found_block = True
            continue
        if stripped.startswith("*"):
            in_nodes = False
            continue
        if in_nodes and stripped:
            parts = stripped.split(",")
            node_id, z = int(parts[0]), float(parts[3])
            if z <= tol:
                base.append(node_id)
    if not found_block:
        raise RuntimeError(f"No hay bloque *NODE en {inp_path}")
    return base


def write_modal_deck(
    mesh_inp: Path,
    deck_path: Path,
    base_nodes: list[int],
    material: Material,
    n_modes: int = 10,
) -> Path:
    """
    Escribe el .inp maestro de CalculiX: incluye la malla, define el NSET de
    la base, material, sección sólida, empotramiento y paso *FREQUENCY.
    El deck debe correrse con cwd en su propia carpeta (el *INCLUDE es relativo).
    """
    lines = [f"*INCLUDE, INPUT={mesh_inp.name}", "*NSET, NSET=BASE"]
    for i in range(0, len(base_nodes), 16):  # máx 16 entradas por línea en ccx
        lines.append(", ".join(str(n) for n in base_nodes[i : i + 16]))
    lines += [
        f"*MATERIAL, NAME={material.name}",
        "*ELASTIC",
        f"{material.E}, {material.nu}",
        "*DENSITY",
        f"{material.rho}",
        f"*SOLID SECTION, ELSET=STRUCTURE, MATERIAL={material.name}",
        "*BOUNDARY",
        "BASE, 1, 3",
        "*STEP",
        "*FREQUENCY",
        f"{n_modes}",
        "*NODE FILE",
        "U",
        "*END STEP",
        "",
    ]
    deck_path = Path(deck_path)
    deck_path.write_text("\n".join(lines))
    return deck_path


# ------------------------------------------------------------------ #
# Resultados                                                           #
# ------------------------------------------------------------------ #

def parse_frequencies(dat_path: Path) -> list[float]:
    """
    Extrae las frecuencias naturales (Hz) de la tabla EIGENVALUE OUTPUT
    de un archivo .dat de CalculiX.

    Formato de fila: MODE_NO  EIGENVALUE  FREQ_RAD  FREQ_HZ  FREQ_IMAG
    """
    freqs: list[float] = []
    for line in Path(dat_path).read_text().splitlines():
        tokens = line.split()
        if len(tokens) == 5 and tokens[0].isdigit():
            try:
                freqs.append(float(tokens[3]))
            except ValueError:
                continue
    if not freqs:
        raise RuntimeError(
            f"No se encontró ninguna frecuencia en {dat_path}. "
            "¿El análisis de CalculiX terminó sin errores?"
        )
    return freqs

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
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


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
# Geometría (FreeCAD → .step)                                          #
# ------------------------------------------------------------------ #

def generate_toy_step(out_path: Path) -> Path:
    """Genera la geometría de prueba ejecutando FreeCADCmd como subprocess."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    script = _SCRIPTS_DIR / "freecad_toy_geometry.py"
    result = subprocess.run(
        [str(find_freecadcmd()), str(script), str(out_path)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"FreeCADCmd falló (código {result.returncode}).\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )
    return out_path


# ------------------------------------------------------------------ #
# Malla → deck de análisis modal                                       #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class MeshInfo:
    n_nodes: int
    n_elements: int


def mesh_step(step_path: Path, inp_path: Path, element_size: float = 0.4) -> MeshInfo:
    """
    Malla un .step con tetraedros cuadráticos (C3D10) y escribe un .inp Abaqus.
    OCCTargetUnit="M" convierte los mm del STEP a metros (unidades SI del modelo).
    """
    import gmsh

    inp_path = Path(inp_path)
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            raise RuntimeError(f"El STEP no contiene sólidos: {step_path}")
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in volumes], name="STRUCTURE")

        gmsh.option.setNumber("Mesh.MeshSizeMin", element_size / 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)  # tets de 10 nodos → C3D10

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        _, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
        info = MeshInfo(
            n_nodes=len(node_tags),
            n_elements=sum(len(t) for t in elem_tags),
        )
        gmsh.write(str(inp_path))
        return info
    finally:
        gmsh.finalize()


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
# Solver                                                               #
# ------------------------------------------------------------------ #

def run_ccx(deck_path: Path) -> Path:
    """
    Ejecuta CalculiX sobre el deck. ccx se invoca sin la extensión .inp y con
    cwd en la carpeta del deck (el *INCLUDE de la malla es relativo).
    """
    deck_path = Path(deck_path)
    result = subprocess.run(
        [str(find_ccx()), "-i", deck_path.stem],
        cwd=deck_path.parent, capture_output=True, text=True, timeout=600,
    )
    dat = deck_path.with_suffix(".dat")
    if result.returncode != 0 or not dat.exists():
        raise RuntimeError(
            f"ccx falló (código {result.returncode}).\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )
    return dat


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

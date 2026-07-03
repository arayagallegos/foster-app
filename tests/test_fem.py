"""tests/test_fem.py — Tests del pipeline FEM (módulo 5)."""
from pathlib import Path

import pytest

from app.modules.fem import (
    MAMPOSTERIA_GENERICA,
    Material,
    extract_base_nodes,
    find_ccx,
    find_freecadcmd,
    parse_frequencies,
    write_modal_deck,
)

# ------------------------------------------------------------------ #
# Task 1: parse_frequencies                                            #
# ------------------------------------------------------------------ #

# Bloque real de un .dat de CalculiX (formato de la tabla de autovalores)
DAT_EJEMPLO = """\
     E I G E N V A L U E   O U T P U T

 MODE NO    EIGENVALUE                     FREQUENCY
                                REAL PART            IMAGINARY PART
                          (RAD/TIME)      (CYCLES/TIME)     (RAD/TIME)

      1    0.1421485E+05    0.1192261E+03    0.1897543E+02    0.000000E+00
      2    0.1421501E+05    0.1192268E+03    0.1897554E+02    0.000000E+00
      3    0.5310071E+05    0.2304359E+03    0.3667496E+02    0.000000E+00
"""


def test_parse_frequencies_extrae_hz_en_orden(tmp_path):
    dat = tmp_path / "modal.dat"
    dat.write_text(DAT_EJEMPLO)
    freqs = parse_frequencies(dat)
    assert freqs == pytest.approx([18.97543, 18.97554, 36.67496], rel=1e-4)


def test_parse_frequencies_archivo_sin_tabla_lanza_error(tmp_path):
    dat = tmp_path / "vacio.dat"
    dat.write_text("sin resultados")
    with pytest.raises(RuntimeError, match="frecuencia"):
        parse_frequencies(dat)


# ------------------------------------------------------------------ #
# Task 2: extract_base_nodes                                           #
# ------------------------------------------------------------------ #

INP_EJEMPLO = """\
*Heading
 malla de prueba
*NODE
1, 0.0, 0.0, 0.0
2, 1.5, 0.0, 0.0
3, 0.0, 1.5, 2.9999999
4, 2.5, 0.0, 1.0e-9
5, 0.0, 2.5, 3.0
*ELEMENT, type=C3D10, ELSET=STRUCTURE
1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5
"""


def test_extract_base_nodes_filtra_z_cero(tmp_path):
    inp = tmp_path / "toy.inp"
    inp.write_text(INP_EJEMPLO)
    assert extract_base_nodes(inp) == [1, 2, 4]


def test_extract_base_nodes_sin_bloque_node_lanza_error(tmp_path):
    inp = tmp_path / "malo.inp"
    inp.write_text("*Heading\n vacio\n")
    with pytest.raises(RuntimeError, match="NODE"):
        extract_base_nodes(inp)


# ------------------------------------------------------------------ #
# Task 3: Material y write_modal_deck                                  #
# ------------------------------------------------------------------ #

def test_write_modal_deck_contiene_cards_esenciales(tmp_path):
    mesh = tmp_path / "toy.inp"
    mesh.write_text("*NODE\n1, 0., 0., 0.\n")
    deck = write_modal_deck(
        mesh_inp=mesh,
        deck_path=tmp_path / "modal.inp",
        base_nodes=[1, 2, 3],
        material=MAMPOSTERIA_GENERICA,
        n_modes=10,
    )
    text = deck.read_text()
    assert "*INCLUDE, INPUT=toy.inp" in text
    assert "*NSET, NSET=BASE" in text
    assert "1, 2, 3" in text
    assert "*ELASTIC" in text and "2000000000.0, 0.2" in text
    assert "*DENSITY" in text and "1800.0" in text
    assert "*SOLID SECTION, ELSET=STRUCTURE, MATERIAL=MAMPOSTERIA" in text
    assert "*BOUNDARY" in text and "BASE, 1, 3" in text
    assert "*FREQUENCY" in text and "\n10\n" in text


def test_write_modal_deck_parte_nset_en_lineas_de_16(tmp_path):
    mesh = tmp_path / "toy.inp"
    mesh.write_text("*NODE\n1, 0., 0., 0.\n")
    deck = write_modal_deck(
        mesh_inp=mesh,
        deck_path=tmp_path / "modal.inp",
        base_nodes=list(range(1, 41)),  # 40 nodos → 3 líneas (16+16+8)
        material=Material(name="TEST", E=1.0, nu=0.3, rho=1.0),
        n_modes=5,
    )
    lines = deck.read_text().splitlines()
    nset_idx = lines.index("*NSET, NSET=BASE")
    assert lines[nset_idx + 1].count(",") == 15   # 16 ids
    assert lines[nset_idx + 3].count(",") == 7    # 8 ids restantes


# ------------------------------------------------------------------ #
# Task 4: localización de herramientas externas                        #
# ------------------------------------------------------------------ #

def test_find_freecadcmd_respeta_variable_de_entorno(tmp_path, monkeypatch):
    fake = tmp_path / "FreeCADCmd.exe"
    fake.write_text("")
    monkeypatch.setenv("FOSTER_FREECADCMD", str(fake))
    assert find_freecadcmd() == fake


def test_find_ccx_env_inexistente_lanza_error(monkeypatch):
    monkeypatch.setenv("FOSTER_CCX", r"C:\no\existe\ccx.exe")
    with pytest.raises(FileNotFoundError, match="FOSTER_CCX"):
        find_ccx()


# ------------------------------------------------------------------ #
# Tasks 5-7: etapas con herramientas externas (skip si no instaladas)  #
# ------------------------------------------------------------------ #

def _has_freecad() -> bool:
    try:
        find_freecadcmd()
        return True
    except FileNotFoundError:
        return False


def _has_ccx() -> bool:
    try:
        find_ccx()
        return True
    except FileNotFoundError:
        return False


@pytest.fixture(scope="session")
def toy_step(tmp_path_factory):
    if not _has_freecad():
        pytest.skip("FreeCAD no instalado")
    from app.modules.fem import generate_toy_step
    return generate_toy_step(tmp_path_factory.mktemp("geom") / "toy.step")


@pytest.mark.skipif(not _has_freecad(), reason="FreeCAD no instalado")
def test_generate_toy_step_crea_archivo(toy_step):
    assert toy_step.exists() and toy_step.stat().st_size > 1_000


@pytest.mark.skipif(not (_has_freecad() and _has_ccx()),
                    reason="requiere FreeCAD y ccx")
def test_pipeline_completo_frecuencias_plausibles(toy_step, tmp_path):
    from app.modules.fem import mesh_step, run_ccx
    mesh_step(toy_step, tmp_path / "toy.inp", element_size=0.6)  # malla gruesa: rápido
    base = extract_base_nodes(tmp_path / "toy.inp")
    deck = write_modal_deck(tmp_path / "toy.inp", tmp_path / "modal.inp",
                            base, MAMPOSTERIA_GENERICA, n_modes=5)
    freqs = parse_frequencies(run_ccx(deck))
    assert len(freqs) == 5
    # Plausibilidad física: estructura de mampostería rígida y baja → entre 1 y 200 Hz
    assert all(1.0 < f < 200.0 for f in freqs)
    assert freqs == sorted(freqs)


@pytest.mark.skipif(not _has_freecad(), reason="FreeCAD no instalado")
def test_mesh_step_genera_inp_c3d10(toy_step, tmp_path):
    from app.modules.fem import mesh_step
    info = mesh_step(toy_step, tmp_path / "toy.inp", element_size=0.5)
    text = (tmp_path / "toy.inp").read_text()
    assert "C3D10" in text
    assert "STRUCTURE" in text.upper()
    assert info.n_nodes > 100 and info.n_elements > 100
    # Debe haber nodos en la base (z=0) para poder empotrar
    base = extract_base_nodes(tmp_path / "toy.inp")
    assert len(base) > 10

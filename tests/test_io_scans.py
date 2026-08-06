"""tests/test_io_scans.py — Carga de scans del e57 con caché."""
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from app.core.io import ScanCacheInfo, _scans_desde_cache


def _escribir_scan_cache(cache_dir: Path, idx: int, n: int) -> None:
    rng = np.random.default_rng(idx)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 5, (n, 3))))
    cache_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(cache_dir / f"scan_{idx:02d}.ply"), pcd)


def test_scans_desde_cache_lee_los_ply(tmp_path):
    _escribir_scan_cache(tmp_path, 0, 2000)
    _escribir_scan_cache(tmp_path, 1, 3000)
    scans = _scans_desde_cache(tmp_path, voxel_grueso=0.10)
    assert len(scans) == 2
    assert all(isinstance(s, ScanCacheInfo) for s in scans)
    assert scans[0].index == 0 and scans[1].index == 1
    assert scans[0].fine_path.exists()
    # el grueso tiene MENOS (o igual) puntos que el fino (voxel down)
    assert len(scans[0].pcd_grueso.points) <= scans[0].n_pts_fino


def test_scans_desde_cache_vacio_lista_vacia(tmp_path):
    assert _scans_desde_cache(tmp_path, voxel_grueso=0.10) == []


def _intentar_escribir_e57(path: Path) -> bool:
    try:
        import pye57
        w = pye57.E57(str(path), mode="w")
        rng = np.random.default_rng(0)
        for _ in range(2):
            xyz = rng.uniform(0, 5, (5000, 3))
            w.write_scan_raw({
                "cartesianX": xyz[:, 0], "cartesianY": xyz[:, 1], "cartesianZ": xyz[:, 2],
            })
        del w
        return True
    except Exception:
        return False


def test_recachear_recorta_a_la_caja(tmp_path):
    """El re-cacheo a resolución fina debe quedarse SOLO con lo que cae dentro de
    la caja indicada. Es lo que permite bajar el vóxel sin que el archivo explote:
    el entorno (que es la mayor parte de los puntos) se descarta al leer."""
    from app.core.io import recachear_e57_recortado

    e57 = tmp_path / "mini.e57"
    if not _intentar_escribir_e57(e57):
        pytest.skip("pye57 no puede escribir .e57 en este entorno")

    # los puntos del fixture están en [0,5]^3; se recorta a un octante
    mn, mx = np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0])
    scans = recachear_e57_recortado(
        str(e57), tmp_path / "cache_fino", mn, mx,
        voxel_fino=0.01, voxel_grueso=0.05,
    )
    assert scans, "no se recuperó ningún scan"
    for s in scans:
        pts = np.asarray(o3d.io.read_point_cloud(str(s.fine_path)).points)
        assert len(pts) > 0
        assert np.all(pts >= mn - 1e-6) and np.all(pts <= mx + 1e-6)


def test_recachear_reusa_el_cache_si_ya_existe(tmp_path):
    """La segunda llamada no debe releer el .e57 (que es la operación cara)."""
    from app.core.io import recachear_e57_recortado

    cache = tmp_path / "cache_fino"
    _escribir_scan_cache(cache, 0, 500)
    scans = recachear_e57_recortado(
        "ruta/inexistente.e57", cache,          # no se toca: el caché ya está
        np.zeros(3), np.full(3, 5.0),
    )
    assert len(scans) == 1


def test_e57_scan_count(tmp_path):
    e57 = tmp_path / "mini.e57"
    if not _intentar_escribir_e57(e57):
        pytest.skip("pye57 no pudo escribir un .e57 de prueba")
    from app.core.io import _e57_scan_count
    assert _e57_scan_count(str(e57)) == 2


def test_load_e57_scans_cached_e2e(tmp_path):
    e57 = tmp_path / "mini.e57"
    if not _intentar_escribir_e57(e57):
        pytest.skip("pye57 no pudo escribir un .e57 de prueba en este entorno")
    from app.core.io import load_e57_scans_cached
    cache = tmp_path / "cache"
    scans = load_e57_scans_cached(str(e57), cache, voxel_fino=0.05, voxel_grueso=0.2)
    assert len(scans) == 2
    assert all(s.fine_path.exists() for s in scans)

    # caché-hit: borrar el .e57 y volver a cargar debe funcionar igual
    e57.unlink()
    scans2 = load_e57_scans_cached(str(e57), cache, voxel_fino=0.05, voxel_grueso=0.2)
    assert len(scans2) == 2

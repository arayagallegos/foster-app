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

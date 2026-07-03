import numpy as np
import open3d as o3d
import pytest

from app.modules.processing import crop_cloud, apply_lasso


def _make_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def test_crop_keeps_points_inside_box():
    pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [5, 5, 5]], dtype=np.float64)
    pcd = _make_cloud(pts)
    result = crop_cloud(pcd, np.array([0, 0, 0]), np.array([2, 2, 2]))
    assert len(result.points) == 3


def test_crop_preserves_colors():
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    colors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    pcd = _make_cloud(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    result = crop_cloud(pcd, np.array([-1, -1, -1]), np.array([0.5, 0.5, 0.5]))
    assert result.has_colors()
    assert len(result.points) == 1


def test_crop_empty_raises():
    pts = np.array([[10, 10, 10]], dtype=np.float64)
    pcd = _make_cloud(pts)
    with pytest.raises(ValueError, match="no contiene puntos"):
        crop_cloud(pcd, np.array([0, 0, 0]), np.array([1, 1, 1]))


def test_apply_lasso_union_selects_points_inside_polygon():
    screen_pts = np.array([[10.0, 10.0], [50.0, 50.0], [200.0, 200.0]], dtype=np.float64)
    valid = np.array([True, True, True])
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    current = np.array([False, False, False])
    result = apply_lasso(screen_pts, valid, polygon, "union", current)
    assert result[0] == True
    assert result[1] == True
    assert result[2] == False


def test_apply_lasso_excludes_points_behind_camera():
    screen_pts = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
    valid = np.array([True, False])
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    current = np.array([False, False])
    result = apply_lasso(screen_pts, valid, polygon, "union", current)
    assert result[0] == True
    assert result[1] == False


def test_apply_lasso_difference_removes_selected_from_current():
    screen_pts = np.array([[10.0, 10.0], [50.0, 50.0], [200.0, 200.0]], dtype=np.float64)
    valid = np.ones(3, dtype=bool)
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    current = np.array([True, True, True])
    result = apply_lasso(screen_pts, valid, polygon, "difference", current)
    assert result[0] == False
    assert result[1] == False
    assert result[2] == True


def test_apply_lasso_intersection_keeps_only_overlap():
    screen_pts = np.array([[10.0, 10.0], [200.0, 200.0]], dtype=np.float64)
    valid = np.ones(2, dtype=bool)
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    current = np.array([True, True])
    result = apply_lasso(screen_pts, valid, polygon, "intersection", current)
    assert result[0] == True
    assert result[1] == False


def test_apply_lasso_symmetric_difference():
    screen_pts = np.array([[10.0, 10.0], [200.0, 200.0]], dtype=np.float64)
    valid = np.ones(2, dtype=bool)
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    current = np.array([False, True])
    result = apply_lasso(screen_pts, valid, polygon, "symmetric_difference", current)
    assert result[0] == True
    assert result[1] == True


def test_apply_lasso_raises_on_fewer_than_3_vertices():
    screen_pts = np.array([[10.0, 10.0]], dtype=np.float64)
    valid = np.ones(1, dtype=bool)
    polygon = [(0, 0), (100, 0)]
    current = np.array([False])
    with pytest.raises(ValueError, match="3 puntos"):
        apply_lasso(screen_pts, valid, polygon, "union", current)


def test_apply_lasso_raises_on_invalid_set_op():
    screen_pts = np.array([[10.0, 10.0]], dtype=np.float64)
    valid = np.ones(1, dtype=bool)
    polygon = [(0, 0), (100, 0), (100, 100)]
    current = np.array([False])
    with pytest.raises(ValueError, match="no valida"):
        apply_lasso(screen_pts, valid, polygon, "xor", current)

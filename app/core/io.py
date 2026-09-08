"""
io.py — Carga y exportación de archivos de nubes de puntos.

Soporta: .e57, .las, .laz, .ply, .pcd
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


# ------------------------------------------------------------------ #
# Función pública principal                                            #
# ------------------------------------------------------------------ #

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Carga una nube de puntos desde disco.
    Detecta el formato por extensión y delega al loader correspondiente.

    Args:
        path: Ruta absoluta al archivo.

    Returns:
        Nube de puntos Open3D lista para usar.

    Raises:
        ValueError: Si el formato no está soportado.
        FileNotFoundError: Si el archivo no existe.
        RuntimeError: Si la carga falla por contenido inválido.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    ext = p.suffix.lower()

    loaders = {
        ".e57": _load_e57,
        ".las":  _load_las,
        ".laz":  _load_las,
        ".ply":  _load_ply,
        ".pcd":  _load_pcd,
    }

    if ext not in loaders:
        supported = ", ".join(loaders.keys())
        raise ValueError(
            f"Formato '{ext}' no soportado. "
            f"Formatos válidos: {supported}"
        )

    pcd = loaders[ext](str(path))

    if len(pcd.points) == 0:
        raise RuntimeError(
            f"El archivo se cargó pero no contiene puntos: {path}"
        )

    return pcd


def load_e57_scans(path: str, voxel_size: float = 0.05) -> list:
    """
    Carga cada scan del e57 por separado (sin combinar) para registro posterior.
    Aplica downsampling ligero a cada scan para que el registro sea manejable.

    Returns:
        Lista de (PointCloud, scan_idx) — uno por scan en el archivo.
    """
    try:
        import pye57
    except ImportError:
        raise ImportError("pip install pye57")

    e57_file = pye57.E57(path)
    n_scans = e57_file.scan_count
    print(f"[io] Cargando {n_scans} scans individuales para registro...")

    scans = []
    for i in range(n_scans):
        print(f"[io]   Scan {i+1}/{n_scans}...", end=" ", flush=True)
        try:
            data = e57_file.read_scan_raw(i)
        except Exception as ex:
            print(f"error: {ex}")
            continue

        if "cartesianX" not in data:
            print("sin coordenadas, omitido.")
            continue

        xyz = np.column_stack([
            np.asarray(data["cartesianX"], dtype=np.float32),
            np.asarray(data["cartesianY"], dtype=np.float32),
            np.asarray(data["cartesianZ"], dtype=np.float32),
        ])
        del data["cartesianX"], data["cartesianY"], data["cartesianZ"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        del xyz

        if "colorRed" in data:
            colors = np.column_stack([
                np.asarray(data["colorRed"],   dtype=np.float32),
                np.asarray(data["colorGreen"], dtype=np.float32),
                np.asarray(data["colorBlue"],  dtype=np.float32),
            ]) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        del data

        pcd_down = pcd.voxel_down_sample(voxel_size)
        del pcd
        print(f"{len(pcd_down.points):,} pts")
        scans.append((pcd_down, i))

    return scans


def export_point_cloud(pcd: o3d.geometry.PointCloud, path: str) -> None:
    """
    Exporta una nube de puntos a disco.
    Formato detectado por extensión (.ply, .pcd).
    """
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


# ------------------------------------------------------------------ #
# Scans del .e57 como capas, con caché de dos resoluciones            #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class ScanCacheInfo:
    index: int
    n_pts_fino: int
    fine_path: Path                        # scan_NN.ply a voxel fino (en disco)
    pcd_grueso: o3d.geometry.PointCloud    # versión gruesa para el visor


def _e57_scan_count(path: str) -> int:
    """Número de scans del .e57 leyendo solo el header (barato)."""
    import pye57
    return int(pye57.E57(str(path)).scan_count)


def _scans_desde_cache(cache_dir: Path, voxel_grueso: float) -> list[ScanCacheInfo]:
    """Carga scans ya cacheados (scan_NN.ply) sin tocar el .e57."""
    cache_dir = Path(cache_dir)
    scans: list[ScanCacheInfo] = []
    for ply in sorted(cache_dir.glob("scan_*.ply")):
        idx = int(ply.stem.split("_")[1])
        fino = o3d.io.read_point_cloud(str(ply))
        n_fino = len(fino.points)
        if n_fino == 0:
            continue
        grueso = fino.voxel_down_sample(voxel_grueso)
        scans.append(ScanCacheInfo(index=idx, n_pts_fino=n_fino,
                                   fine_path=ply, pcd_grueso=grueso))
    return scans


def _leer_e57_a_cache(path, cache_dir, voxel_fino, voxel_grueso, progress_cb,
                      bounds=None):
    """
    Streaming del .e57: por cada scan → voxel fino → escribe scan_NN.ply en cache_dir,
    y genera la versión gruesa para el visor. float32 y `del` agresivo para no acumular
    memoria (mismo patrón que _load_e57).

    `bounds` = (min_bound, max_bound) opcional, en coordenadas mundo. Si se entrega,
    cada scan se RECORTA a esa caja antes de voxelizar. Sirve para releer el archivo
    a resolución fina una vez que el usuario ya delimitó la estructura de interés:
    antes del recorte se estaría guardando todo el entorno (terreno, árboles,
    edificios vecinos), que es la mayor parte de los puntos y se descarta igual.

    Nota: se usa `read_scan` (no `read_scan_raw`), que ya devuelve los puntos en
    coordenadas mundo con la pose del scan aplicada.
    """
    # Se valida antes de abrir el archivo, porque leerlo entero para descubrir
    # a mitad de camino que la caja llegó vacía cuesta minutos.
    if bounds is not None and (bounds[0] is None or bounds[1] is None):
        raise ValueError(
            "Se pidió recortar por una caja pero uno de sus límites llegó vacío.")
    try:
        import pye57
    except ImportError:
        raise ImportError("Instala pye57 para leer archivos .e57: pip install pye57")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    e57 = pye57.E57(str(path))
    n_scans = e57.scan_count
    scans: list[ScanCacheInfo] = []

    for i in range(n_scans):
        if progress_cb:
            progress_cb(int(100 * i / max(n_scans, 1)), f"Leyendo scan {i+1}/{n_scans}...")
        try:
            data = e57.read_scan(i, colors=True, ignore_missing_fields=True)
        except Exception:
            continue
        if "cartesianX" not in data:
            continue

        xyz = np.column_stack([
            np.asarray(data["cartesianX"], dtype=np.float32),
            np.asarray(data["cartesianY"], dtype=np.float32),
            np.asarray(data["cartesianZ"], dtype=np.float32),
        ])
        del data["cartesianX"], data["cartesianY"], data["cartesianZ"]

        # Recorte a la caja ANTES de voxelizar: descarta el grueso de los puntos
        # (el entorno) y deja solo la estructura, que es lo que justifica pagar
        # una resolución más fina.
        dentro = None
        if bounds is not None:
            mn, mx = (np.asarray(bounds[0], dtype=float),
                      np.asarray(bounds[1], dtype=float))
            dentro = np.all((xyz >= mn) & (xyz <= mx), axis=1)
            xyz = xyz[dentro]
            if len(xyz) == 0:
                del xyz, data
                continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        del xyz
        if "colorRed" in data and "colorGreen" in data and "colorBlue" in data:
            cols = np.column_stack([
                np.asarray(data["colorRed"], dtype=np.float32),
                np.asarray(data["colorGreen"], dtype=np.float32),
                np.asarray(data["colorBlue"], dtype=np.float32),
            ]) / 255.0
            if dentro is not None:      # mismo recorte que los puntos
                cols = cols[dentro]
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
            del cols
        del data

        fino = pcd.voxel_down_sample(voxel_fino)
        del pcd
        if len(fino.points) == 0:
            continue
        fine_path = cache_dir / f"scan_{i:02d}.ply"
        o3d.io.write_point_cloud(str(fine_path), fino)
        grueso = fino.voxel_down_sample(voxel_grueso)
        scans.append(ScanCacheInfo(index=i, n_pts_fino=len(fino.points),
                                   fine_path=fine_path, pcd_grueso=grueso))
        del fino

    if progress_cb:
        progress_cb(100, f"{len(scans)} scans cargados.")
    return scans


def load_e57_scans_cached(
    path: str,
    cache_dir: Path,
    voxel_fino: float = 0.03,
    voxel_grueso: float = 0.10,
    progress_cb=None,
) -> list[ScanCacheInfo]:
    """
    Carga cada scan del .e57 como capa, con caché de dos resoluciones.
    1ª vez: streaming del .e57 → escribe scan_NN.ply (voxel fino) en cache_dir.
    Siguientes: si el caché existe, lee de ahí sin tocar el .e57.
    """
    cache_dir = Path(cache_dir)
    if cache_dir.exists() and any(cache_dir.glob("scan_*.ply")):
        if progress_cb:
            progress_cb(100, "Cargando scans desde caché...")
        return _scans_desde_cache(cache_dir, voxel_grueso)
    return _leer_e57_a_cache(path, cache_dir, voxel_fino, voxel_grueso, progress_cb)


def recachear_e57_recortado(
    path: str,
    cache_dir: Path,
    min_bound,
    max_bound,
    voxel_fino: float = 0.01,
    voxel_grueso: float = 0.03,
    progress_cb=None,
) -> list[ScanCacheInfo]:
    """
    Relee el .e57 a resolución fina, quedándose solo con lo que cae dentro de la
    caja [min_bound, max_bound] (coordenadas mundo).

    Por qué existe: el caché inicial se construye con un vóxel grueso porque debe
    cubrir toda la escena capturada (terreno, vegetación, construcciones vecinas),
    donde está la mayor parte de los puntos. Una vez que el usuario delimitó la
    estructura de interés, se puede pagar una resolución bastante más fina, porque
    solo se guarda esa región. La resolución importa: el vóxel debe ser bastante
    menor que los espesores que se quieran distinguir.

    Escribe en un cache_dir NUEVO (no pisa el original), para poder volver atrás.
    """
    cache_dir = Path(cache_dir)
    if cache_dir.exists() and any(cache_dir.glob("scan_*.ply")):
        if progress_cb:
            progress_cb(100, "Cargando scans recortados desde caché...")
        return _scans_desde_cache(cache_dir, voxel_grueso)
    return _leer_e57_a_cache(path, cache_dir, voxel_fino, voxel_grueso, progress_cb,
                             bounds=(min_bound, max_bound))


# ------------------------------------------------------------------ #
# Loaders por formato                                                  #
# ------------------------------------------------------------------ #

def _load_ply(path: str) -> o3d.geometry.PointCloud:
    """Carga un archivo .ply (salida típica de Metashape/RealityCapture)."""
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def _load_pcd(path: str) -> o3d.geometry.PointCloud:
    """Carga un archivo .pcd (formato Open3D nativo)."""
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def _load_las(path: str) -> o3d.geometry.PointCloud:
    """
    Carga un archivo .las o .laz (formato LiDAR estándar).
    Extrae XYZ y color RGB si está disponible.
    """
    try:
        import laspy
    except ImportError:
        raise ImportError(
            "Instala laspy para leer archivos .las/.laz:\n"
            "pip install laspy[lazrs]"
        )

    las = laspy.read(path)
    xyz = np.column_stack([
        las.x.scaled_array(),
        las.y.scaled_array(),
        las.z.scaled_array(),
    ])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Intentar extraer colores RGB (no todos los .las los tienen)
    try:
        r = np.asarray(las.red,   dtype=np.float64)
        g = np.asarray(las.green, dtype=np.float64)
        b = np.asarray(las.blue,  dtype=np.float64)

        # LAS almacena colores en rango 0-65535, normalizar a 0-1
        max_val = r.max() if r.max() > 0 else 1.0
        if max_val > 255:
            r, g, b = r / 65535.0, g / 65535.0, b / 65535.0
        else:
            r, g, b = r / 255.0,   g / 255.0,   b / 255.0

        colors = np.column_stack([r, g, b])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    except Exception:
        # Sin color: se coloreará por altura en el viewer
        pass

    return pcd


def _load_e57(path: str, voxel_size: float = 0.03) -> o3d.geometry.PointCloud:
    """
    Carga un archivo .e57 grande de forma streaming: lee cada scan por separado,
    aplica voxel downsampling inmediatamente y libera la memoria antes del siguiente.

    Para un archivo de 19 GB esto es crítico: nunca se carga todo en RAM.
    El resultado es una nube downsampleada lista para visualización.

    Args:
        path:       Ruta al archivo .e57
        voxel_size: Tamaño de voxel en metros para downsampling por scan (default 3cm).
                    Ajustar según el tamaño del edificio y la RAM disponible:
                    - 0.02 (2cm): mayor detalle, más RAM
                    - 0.03 (3cm): buen equilibrio para edificios pequeños
                    - 0.05 (5cm): menos detalle, menos RAM
    """
    try:
        import pye57
    except ImportError:
        raise ImportError(
            "Instala pye57 para leer archivos .e57:\n"
            "pip install pye57"
        )

    e57_file = pye57.E57(path)
    n_scans = e57_file.scan_count
    print(f"[io] .e57: {n_scans} scan(s) detectados. Leyendo con voxel={voxel_size}m...")

    accumulated_pts    = []
    accumulated_colors = []
    total_raw          = 0
    total_kept         = 0
    has_color          = False

    for scan_idx in range(n_scans):
        print(f"[io]   Scan {scan_idx + 1}/{n_scans}...", end=" ", flush=True)
        # read_scan filtra puntos inválidos (cartesianInvalidState/sphericalInvalidState)
        # y aplica la transformación de pose local→global automáticamente.
        # También convierte coordenadas esféricas a cartesianas si es necesario.
        try:
            data = e57_file.read_scan(scan_idx, colors=True, ignore_missing_fields=True)
        except Exception as ex:
            print(f"error: {ex}")
            continue

        if "cartesianX" not in data:
            print("sin coordenadas válidas, omitido.")
            continue

        n_pts = len(data["cartesianX"])
        total_raw += n_pts

        # --- float32: mitad de memoria que float64 ---
        xyz = np.column_stack([
            np.asarray(data["cartesianX"], dtype=np.float32),
            np.asarray(data["cartesianY"], dtype=np.float32),
            np.asarray(data["cartesianZ"], dtype=np.float32),
        ])

        print(
            f"\n[scan {scan_idx}]",
            f"X[{xyz[:,0].min():.3f},{xyz[:,0].max():.3f}]",
            f"Y[{xyz[:,1].min():.3f},{xyz[:,1].max():.3f}]",
            f"Z[{xyz[:,2].min():.3f},{xyz[:,2].max():.3f}]"
        )
        # Liberar data de este scan inmediatamente
        del data["cartesianX"], data["cartesianY"], data["cartesianZ"]

        # Construir nube Open3D del scan
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        del xyz  # liberar

        # Color si existe
        if "colorRed" in data and "colorGreen" in data and "colorBlue" in data:
            colors = np.column_stack([
                np.asarray(data["colorRed"],   dtype=np.float32),
                np.asarray(data["colorGreen"], dtype=np.float32),
                np.asarray(data["colorBlue"],  dtype=np.float32),
            ]) / 255.0
            scan_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            del colors
            has_color = True

        del data  # liberar todo el dict del scan

        # --- Downsample inmediato: aquí está la clave ---
        scan_down = scan_pcd.voxel_down_sample(voxel_size)
        del scan_pcd  # liberar la nube full de este scan

        n_down = len(scan_down.points)
        total_kept += n_down
        print(f"{n_pts:,} pts → {n_down:,} después de voxel")

        accumulated_pts.append(np.asarray(scan_down.points, dtype=np.float32))
        if has_color and scan_down.has_colors():
            accumulated_colors.append(
                np.asarray(scan_down.colors, dtype=np.float32)
            )
        del scan_down

    if not accumulated_pts:
        raise RuntimeError("El archivo .e57 no contiene scans con coordenadas válidas.")

    print(
        f"[io] Carga completa: {total_raw:,} pts originales → "
        f"{total_kept:,} pts tras downsampling ({voxel_size*100:.0f}cm voxel)"
    )

    # Combinar todos los scans downsampleados
    all_pts = np.vstack(accumulated_pts).astype(np.float64)
    del accumulated_pts

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(all_pts)
    del all_pts

    if has_color and accumulated_colors:
        all_colors = np.vstack(accumulated_colors).astype(np.float64)
        result.colors = o3d.utility.Vector3dVector(np.clip(all_colors, 0, 1))
        del accumulated_colors, all_colors

    # Downsample final para eliminar solapamiento entre scans
    final = result.voxel_down_sample(voxel_size)

    print(f"[io] Nube final tras merge: {len(final.points):,} puntos")

    bbox = final.get_axis_aligned_bounding_box()

    print("\n========== DEBUG NUBE ==========")
    print("Min:", bbox.min_bound)
    print("Max:", bbox.max_bound)
    print("Extent:", bbox.get_extent())
    print("Center:", bbox.get_center())

    pts = np.asarray(final.points)

    print("Primeros 10 puntos:")
    print(pts[:10])

    print(
        "Rangos XYZ:",
        pts[:, 0].min(), pts[:, 0].max(),
        pts[:, 1].min(), pts[:, 1].max(),
        pts[:, 2].min(), pts[:, 2].max()
    )

    print("================================\n")

    return final


# ------------------------------------------------------------------ #
# Utilidades                                                           #
# ------------------------------------------------------------------ #

SUPPORTED_EXTENSIONS = [".e57", ".las", ".laz", ".ply", ".pcd"]

def caras_en_ply(path) -> tuple[int, int]:
    """Cuántas caras y vértices declara un `.ply`, leyendo solo su cabecera.

    La herramienta trabaja con nubes de puntos, y al abrir un `.ply` que contiene
    una malla el lector se queda solo con sus vértices y descarta las caras sin
    avisar. Saberlo de antemano permite advertirlo.

    Se lee la cabecera y no el archivo, porque una malla grande tarda en cargarse
    y aquí solo hacen falta dos números. Devuelve (0, 0) si no es un `.ply` o si
    la cabecera no se puede interpretar.
    """
    ruta = Path(path)
    if ruta.suffix.lower() != ".ply":
        return 0, 0
    caras = vertices = 0
    try:
        with open(ruta, "rb") as f:
            for _ in range(200):          # la cabecera nunca es tan larga
                linea = f.readline()
                if not linea:
                    break
                texto = linea.decode("ascii", errors="ignore").strip().lower()
                if texto.startswith("end_header"):
                    break
                if texto.startswith("element face"):
                    caras = int(texto.split()[2])
                elif texto.startswith("element vertex"):
                    vertices = int(texto.split()[2])
    except (OSError, ValueError, IndexError):
        return 0, 0
    return caras, vertices


FILTER_STRING = (
    "Nubes de puntos ("
    + " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
    + ");;"
    + "E57 (*.e57);;"
    + "LAS/LAZ (*.las *.laz);;"
    + "PLY (*.ply);;"
    + "PCD (*.pcd);;"
    + "Todos los archivos (*)"
)

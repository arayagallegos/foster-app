"""
segment_circles.py — Perfil de radio (exterior e interior) por altura.

Corta la nube en bandas horizontales y ajusta DOS círculos concéntricos por banda
(cara exterior + cara interior), para obtener r_ext(z), r_int(z) y el espesor(z) del
muro, cornisa, faldón y cúpula. Usa `perfil_radios` del módulo de segmentación.
Enfoque de secciones horizontales à la Funari et al.
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.modules.segmentation import perfil_radios

pcd = o3d.io.read_point_cloud(
    r"C:\Users\rodri\Desktop\Universidad\Memoria\OBS FOSTER\exports\recorte_final.ply")
pcd = pcd.voxel_down_sample(0.03)          # 3 cm: fino para distinguir las dos caras
pts = np.asarray(pcd.points)
pts = pts[pts[:, 2] > 811.2]               # fuera el suelo

bandas = perfil_radios(pts, paso=0.10, eps=0.03)

# ---- tabla ----
print(f"{'z':>8} {'r_ext':>8} {'r_int':>8} {'espesor':>8}")
for b in bandas:
    if b.r_int is not None:
        print(f"{b.z:8.2f} {b.r_ext:8.3f} {b.r_int:8.3f} {b.r_ext - b.r_int:8.3f}")
    else:
        print(f"{b.z:8.2f} {b.r_ext:8.3f} {'-':>8} {'-':>8}")

# ---- gráfico ----
z_ext = [b.z for b in bandas]
radios_ext = [b.r_ext for b in bandas]
z_int = [b.z for b in bandas if b.r_int is not None]
radios_int = [b.r_int for b in bandas if b.r_int is not None]

plt.figure(figsize=(6, 8))
plt.plot(radios_ext, z_ext, "o-", label="cara exterior", color="tab:blue")
plt.plot(radios_int, z_int, "s-", label="cara interior", color="tab:red")
plt.xlabel("radio del círculo [m]")
plt.ylabel("altura z [m]")
plt.title("Perfil de radio exterior/interior — Observatorio Foster")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/perfil_radios.png", dpi=120)
print("\nGuardado: output/perfil_radios.png")

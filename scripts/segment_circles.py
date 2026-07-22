"""
segment_circles.py — Perfil de radio (exterior e interior) por altura.

Corta la nube en bandas horizontales y ajusta DOS círculos concéntricos por banda
(cara exterior + cara interior), para obtener r_ext(z), r_int(z) y el espesor(z) del
muro, cornisa, faldón y cúpula. Enfoque de secciones horizontales à la Funari et al.
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.modules.segmentation import fit_circle_ransac

pcd = o3d.io.read_point_cloud(
    r"C:\Users\rodri\Desktop\Universidad\Memoria\OBS FOSTER\exports\recorte_final.ply")
pcd = pcd.voxel_down_sample(0.03)          # 3 cm: fino para distinguir las dos caras
pts = np.asarray(pcd.points)
pts = pts[pts[:, 2] > 811.2]               # fuera el suelo

paso = 0.10
z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

# perfil exterior (todas las bandas) e interior (solo donde hay cara interior)
z_ext, radios_ext = [], []
z_int, radios_int, espesores = [], [], []

z = z_min
while z < z_max:
    banda = pts[(pts[:, 2] >= z) & (pts[:, 2] < z + paso)]
    if len(banda) >= 100:
        try:
            cx1, cy1, r1, m1 = fit_circle_ransac(banda[:, :2], eps=0.03, n_iters=500)
        except RuntimeError:
            z += paso
            continue
        if m1.sum() >= 200 and r1 < 6.0:
            zc = z + paso / 2
            # segundo círculo sobre los puntos que NO son la primera cara
            resto = banda[~m1]
            if len(resto) >= 100:
                try:
                    cx2, cy2, r2, m2 = fit_circle_ransac(resto[:, :2], eps=0.03, n_iters=500)
                    concentrico = np.hypot(cx2 - cx1, cy2 - cy1) < 0.3
                    r_out, r_in = max(r1, r2), min(r1, r2)
                    esp = r_out - r_in
                    if concentrico and m2.sum() >= 150 and 0.02 < esp < 1.0:
                        z_int.append(zc)
                        radios_int.append(r_in)
                        espesores.append(esp)
                        z_ext.append(zc)
                        radios_ext.append(r_out)   # el mayor es el exterior
                        z += paso
                        continue
                except RuntimeError:
                    pass
            # sin cara interior válida: registrar solo el exterior
            z_ext.append(zc)
            radios_ext.append(r1)
    z += paso

# ---- tabla ----
print(f"{'z':>8} {'r_ext':>8} {'r_int':>8} {'espesor':>8}")
int_por_z = {z: (ri, e) for z, ri, e in zip(z_int, radios_int, espesores)}
for z, re in zip(z_ext, radios_ext):
    if z in int_por_z:
        ri, e = int_por_z[z]
        print(f"{z:8.2f} {re:8.3f} {ri:8.3f} {e:8.3f}")
    else:
        print(f"{z:8.2f} {re:8.3f} {'-':>8} {'-':>8}")

# ---- gráfico ----
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

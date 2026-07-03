# Foster App — Herramienta Scan-to-FEM para el Observatorio Foster

Aplicación de escritorio para procesamiento de nubes de puntos LiDAR y fotogrametría,
reconstrucción de malla 3D, y exportación a modelos de elementos finitos.

Desarrollada como parte de la tesis de Ingeniería Civil en Computación,
Laboratorio de Patrimonio, Universidad de Chile.

---

## Instalación

### 1. Requisitos
- Python 3.11 o superior
- pip actualizado

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Si tienes problemas con `pye57` en Windows, instala primero:
> `pip install wheel` y luego `pip install pye57`

### 4. Ejecutar

```bash
python main.py
```

---

## Formatos soportados

| Formato | Extensión | Descripción |
|---|---|---|
| E57 | `.e57` | Formato nativo de escáneres FARO, Leica |
| LAS / LAZ | `.las`, `.laz` | Formato estándar LiDAR |
| PLY | `.ply` | Polygon File Format, salida de fotogrametría |
| PCD | `.pcd` | Point Cloud Data format |

---

## Módulos (estado de desarrollo)

- [x] **Módulo 0** — Infraestructura base: viewer 3D, carga de archivos
- [ ] **Módulo 1** — Fusión LiDAR + Fotogrametría (ICP)
- [ ] **Módulo 2** — Procesamiento: filtrado, subsampling, edición
- [ ] **Módulo 3** — Reconstrucción de malla (Poisson)
- [ ] **Módulo 4** — Segmentación (RANSAC + DBSCAN + manual)
- [ ] **Módulo 5** — Exportación FEM (gmsh + meshio)

---

## Estructura del proyecto

```
foster_app/
├── main.py                  # Punto de entrada
├── requirements.txt
├── README.md
└── app/
    ├── core/
    │   ├── project.py       # Estado global de la aplicación
    │   ├── io.py            # Carga y exportación de archivos
    │   └── workers.py       # Hilos para operaciones pesadas
    ├── gui/
    │   ├── main_window.py   # Ventana principal
    │   ├── viewer.py        # Widget 3D interactivo
    │   └── panels/
    │       └── info_panel.py  # Panel de información de la nube
    └── modules/
        ├── fusion.py        # Fusión de nubes
        ├── processing.py    # Filtrado y subsampling
        ├── meshing.py       # Reconstrucción de malla
        ├── segmentation.py  # Segmentación de elementos
        └── fem.py           # Exportación FEM
```

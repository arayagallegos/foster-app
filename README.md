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

## Librerías y su rol

| Librería | Rol en el proyecto | Usada en |
|---|---|---|
| **PyQt6** | Framework de interfaz gráfica (ventana, botones, menús, paneles) | `app/gui/` |
| **PyVista** | Visualización 3D interactiva de la nube (envoltorio amigable de VTK) | `app/gui/viewer.py` |
| **pyvistaqt** | Pegamento: incrusta el visor 3D de PyVista dentro de una ventana PyQt6 | `app/gui/viewer.py` |
| **Open3D** | Procesamiento de nubes de puntos: carga, downsampling, RANSAC de plano, I/O de `.ply` | `core/io.py`, `modules/segmentation.py`, `modules/processing.py` |
| **NumPy** | Base del cálculo numérico (arreglos/matrices); la usan casi todas las demás por debajo | transversal |
| **SciPy** | Herramientas científicas sobre NumPy (álgebra lineal, optimización, espacial) | apoyo numérico |
| **laspy** | Lectura/escritura del formato LiDAR LAS/LAZ | `core/io.py` |
| **pye57** | Lectura del formato E57 scan-por-scan (permite abrir el archivo de 25 GB sin cargarlo entero) | `core/io.py` |
| **gmsh** | Mallado de elementos finitos: lee el `.step` y genera la malla tetraédrica (C3D10) | `modules/fem.py` |
| **matplotlib** | Solo `matplotlib.path.Path`: prueba de punto-en-polígono para la selección por lazo | `modules/processing.py` |

**Herramientas externas (no son librerías pip, se instalan aparte):**

| Herramienta | Rol | Cómo se invoca |
|---|---|---|
| **FreeCAD** | Genera la geometría sólida paramétrica y la exporta a `.step` | `FreeCADCmd.exe` como subproceso (usa su propio intérprete Python, no el venv) |
| **CalculiX** | Solver FEM open source (análisis modal); compatible con el formato Abaqus | `ccx.exe` como subproceso (viene en el `bin` de FreeCAD) |

---

## Módulos (estado de desarrollo)

- [x] **Módulo 0** — Infraestructura base: viewer 3D, carga de archivos
- [ ] **Módulo 1** — Fusión LiDAR + Fotogrametría (ICP)
- [~] **Módulo 2** — Procesamiento: recorte iterativo por capas (caja + lazo, preview verde/rojo, exportación de capas visibles); falta filtrado
- [ ] **Módulo 3** — Reconstrucción de malla (Poisson)
- [~] **Módulo 4** — Segmentación: RANSAC propio funcionando (suelo/tambor/cúpula + espesor de muro, validado con nube sintética); falta corrida real y DL
- [~] **Módulo 5** — Exportación FEM: pipeline modal de prueba funcionando (FreeCAD→gmsh→CalculiX); falta geometría real

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

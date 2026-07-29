# Compilación del puente CGAL (Windows)

> **Por qué existe este documento:** compilar CGAL es el paso que más problemas da a
> quien retome el proyecto. Está escrito para seguirlo al pie de la letra sin conocer
> C++ ni CMake.

El resultado de esta guía es un archivo **`cgal_bridge.cp311-win_amd64.pyd`** en esta
misma carpeta, que Python importa como `import cgal_bridge`.

---

## 0. Requisitos previos

| Requisito | Cómo obtenerlo | Verificar |
|---|---|---|
| **Compilador C++** — basta **Visual Studio Build Tools 2022** (no hace falta el IDE completo) con el componente "Herramientas de compilación de C++" | [visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads/) → "Herramientas de compilación" | `vswhere -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64` devuelve una ruta |
| **CMake ≥ 3.20** | **Ya viene incluido** con los Build Tools, en `…\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin` | `cmake --version` |
| **Git** | [git-scm.com](https://git-scm.com/) | `git --version` |
| **pybind11** (en el venv del proyecto) | `pip install pybind11` | `python -m pybind11 --cmakedir` |

> ⚠️ Instala el compilador **antes** que vcpkg: vcpkg lo necesita para construir CGAL.

> 💡 **CMake no está en el PATH** por defecto al venir con los Build Tools. O lo agregas,
> o usas la ruta completa. Para agregarlo solo en la sesión actual de PowerShell:
> ```powershell
> $env:Path += ";C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
> ```

---

## 1. Instalar vcpkg (gestor de dependencias C++)

Desde PowerShell, en una carpeta **fuera** del proyecto (p. ej. `C:\dev`):

```powershell
git clone https://github.com/microsoft/vcpkg C:\dev\vcpkg
C:\dev\vcpkg\bootstrap-vcpkg.bat
```

## 2. Instalar CGAL y Eigen

```powershell
C:\dev\vcpkg\vcpkg.exe install cgal:x64-windows eigen3:x64-windows
```

> ⏱️ **Esto demora bastante** (30–90 min la primera vez): compila CGAL y sus
> dependencias (Boost, GMP, MPFR). Es normal. Déjalo correr.

## 3. Configurar y compilar el módulo

Desde la raíz del proyecto, con el **venv activo**:

```powershell
cmake -B cgal_bridge/build -S cgal_bridge -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
```

```powershell
cmake --build cgal_bridge/build --config Release
```

Si todo va bien, aparece **`cgal_bridge.cp311-win_amd64.pyd`** en `cgal_bridge/`.

## 4. Verificar

```powershell
python cgal_bridge/test_cgal_bridge.py
```

Debe imprimir que detectó un hoyo, lo rellenó, y que la malla quedó **cerrada**.

---

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `Could not find CGAL` | Falta `-DCMAKE_TOOLCHAIN_FILE=...vcpkg.cmake` en el paso 3, o la ruta a vcpkg está mal. |
| `fatal error C1128: number of sections exceeded` | Falta `/bigobj` — ya está en el `CMakeLists.txt`; asegúrate de no haberlo editado. |
| `ImportError: DLL load failed` al importar | El `.pyd` se compiló contra **otra versión de Python**. Compila con el venv activo y verifica que `Python_EXECUTABLE` apunte al venv. |
| `Eigen3 NO encontrado` (warning) | Solo desactiva el suavizado del parche; los hoyos igual se cierran. Para habilitarlo: `vcpkg install eigen3:x64-windows`. |
| Compila en Debug y falla al importar | Usa siempre `--config Release`: mezclar Debug de C++ con Python Release rompe el enlace en Windows. |
| `cl` no se reconoce | Ejecuta desde el **"Developer PowerShell for VS 2022"**, no desde PowerShell normal. |
| `repair_polygon_soup: no es un miembro de CGAL::Polygon_mesh_processing` | Falta su header propio: `#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>` (no está en `repair.h`). |
| `Warning: header <CGAL/Polygon_mesh_processing/border.h> is deprecated` | En **CGAL 6** se movió: usa `#include <CGAL/boost/graph/border.h>`. |
| `extract_boundary_cycles: no es un miembro de CGAL::Polygon_mesh_processing` | En **CGAL 6** cambió de namespace junto con el header: es `CGAL::extract_boundary_cycles`, no `PMP::`. |

> 💡 **Cómo diagnosticar este tipo de error:** si un símbolo "no es miembro" del namespace,
> búscalo en los headers instalados para ver dónde quedó realmente. Por ejemplo:
> ```powershell
> Select-String -Path "C:\dev\vcpkg\installed\x64-windows\include\CGAL\boost\graph\border.h" -Pattern "namespace|extract_boundary_cycles"
> ```
| Errores en `triangulate_refine_and_fair_hole` por número de argumentos | **CGAL 6** pasó los iteradores de salida a parámetros con nombre. La solución simple: no pasarlos (son opcionales). |

---

## Qué expone el módulo

```python
import cgal_bridge

# Diagnóstico de una malla (V: (n,3) float64, F: (m,3) int32)
stats = cgal_bridge.mesh_stats(V, F)
# -> {'n_vertices','n_faces','n_holes','is_closed','self_intersects','is_outward'}

# Rellenar todos los hoyos
V2, F2 = cgal_bridge.fill_holes(V, F, fair=True)
```

Para agregar más funciones de CGAL: añádelas en `src/cgal_bridge.cpp` (bloque
`PYBIND11_MODULE` al final) y recompila con el paso 3.

# Foster App

Aplicación de escritorio que recorre el tramo que va desde la nube de puntos de
un levantamiento hasta un conjunto de entidades estructurales separadas,
reparadas y malladas por separado, cada una acompañada de una medida de su
fidelidad respecto de los datos medidos.

Ese tramo es el que las herramientas disponibles no cubren. Un laboratorio de
patrimonio dispone de programas para capturar, alinear, visualizar y modelar a
mano, y lo que falta es el paso intermedio: convertir una nube donde todos los
elementos están en contacto y sin distinción entre ellos en partes reconocibles
sobre las que se pueda trabajar.

Desarrollada sobre el levantamiento del Observatorio Manuel Foster (cerro San
Cristóbal, Santiago), en el marco de una memoria de Ingeniería Civil en
Computación de la Universidad de Chile.

---

## El principio que ordena todo

**El usuario aporta la intuición y el algoritmo la precisión.** Reconocer que una
superficie es una cúpula, decidir cuál de las esferas posibles interesa o saber
que una construcción es simétrica son juicios inmediatos para quien mira la
escena y difíciles de automatizar con fiabilidad. Ajustar esa esfera al
milímetro o afinar un plano de simetría es exactamente lo contrario.

Todas las herramientas siguen el mismo ciclo de cuatro pasos: **colocar,
ajustar, previsualizar y confirmar**. El usuario sitúa un manipulador de forma
aproximada, el sistema lo ajusta a los datos al soltarlo, la escena se repinta
distinguiendo lo que quedaría capturado, y solo un gesto explícito compromete la
operación.

Dos criterios atraviesan el diseño:

- **Ninguna operación destruye información.** En lugar de eliminar puntos se
  parte la capa en dos y se conserva la procedencia de cada uno.
- **Lo generado no se mezcla con lo medido.** El material que la herramienta
  fabrica nace en una capa separada, y su incorporación es una decisión
  explícita del usuario, siempre acompañada de una cifra que dice cuánto
  respaldo tiene en mediciones reales.

---

## El flujo

Cuatro etapas encadenadas, cada una con su estado del dato. Ninguna obliga a
llegar a la siguiente: cada estado puede exportarse y reabrirse como punto de
partida.

**1. Preparación y limpieza.** Entra el archivo del levantamiento, sale una nube
acotada a la estructura, separada por escaneo de procedencia y disponible en dos
resoluciones. Cada escaneo entra como una capa. Los recortes por caja y por lazo
se aplican a todas las capas a la vez.

**2. Segmentación.** Entra la nube acotada, salen entidades con nombre. Cuatro
primitivas (esfera, plano, cilindro y cono) que el usuario coloca y el sistema
ajusta por RANSAC, cada una recortable a la porción que interesa. Para lo que no
tiene forma analítica, agrupación por densidad con DBSCAN.

**3. Reparación por simetría.** Entra una entidad con zonas sin cobertura, sale
acompañada del material que la completa. El usuario coloca el plano de simetría
a ojo y el sistema lo refina; el relleno se obtiene trasladando material
efectivamente medido a su posición simétrica, nunca generándolo desde un modelo
ideal.

**4. Reconstrucción.** Entra una entidad, sale una malla de superficie en `.ply`
con sus medidas de fidelidad. Advancing Front usando como vértices los puntos
medidos, con un límite de perímetro de triángulo que impide puentear los vacíos.

### Lo que queda fuera

- **No genera malla de volumen ni ejecuta ninguna simulación.** La salida es el
  insumo de un análisis por elementos finitos, no su resultado.
- **No cose las mallas de entidades adyacentes.** Se generan de forma
  independiente y no comparten vértices en su frontera.
- **No integra fotogrametría con el escaneo terrestre**, ni alinea escaneos: se
  asume que el archivo trae la pose de cada uno ya aplicada.
- Las capacidades de segmentación y reparación se apoyan en superficies de
  revolución y simetrías marcadas. Sobre una construcción irregular el trabajo
  recae en los recortes y en la agrupación por densidad, que son de propósito
  general.

---

## Instalación

Requiere **Python 3.11 o superior**.

```bash
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Si `pye57` falla al instalarse en Windows, instala `wheel` primero y reintenta.

### Ejecutar

```bash
python main.py
```

### Pruebas

```bash
pytest
```

Los módulos de cálculo se ejercitan sin abrir ninguna ventana, que es el
propósito de la separación en capas descrita más abajo. Las pruebas de interfaz
sí necesitan un servidor gráfico: en Linux sin pantalla, `xvfb-run pytest`.

### El puente CGAL

La reconstrucción de superficies es la única capacidad sin equivalente maduro en
el ecosistema de Python, así que se toma de CGAL, escrita en C++. El puente es
un módulo binario aparte construido con `pybind11`, que publica siete funciones
(reconstrucción, cierre de contornos y diagnóstico de mallas) y traduce los
datos entre los dos lenguajes sin copiarlos.

**La aplicación arranca y funciona sin él**, con la herramienta de mallado
deshabilitada y un mensaje que lo explica. Para compilarlo, la guía completa con
su tabla de errores frecuentes está en
[`cgal_bridge/COMPILACION.md`](cgal_bridge/COMPILACION.md). En resumen:

```bash
cmake -B cgal_bridge/build -S cgal_bridge -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
```

```bash
cmake --build cgal_bridge/build --config Release
```

El resultado es un `.pyd` en `cgal_bridge/`, que `meshing.py` importa añadiendo
esa carpeta al camino de búsqueda.

---

## Formatos

| Formato | Extensión | Notas |
|---|---|---|
| E57 | `.e57` | El único que conserva la organización por escaneos separados y la pose de cada uno. Se lee escaneo por escaneo |
| LAS / LAZ | `.las`, `.laz` | Lista plana de puntos, se carga como una sola nube |
| PLY | `.ply` | Entrada y salida. Es el formato en que se exportan nubes y mallas |
| PCD | `.pcd` | Lista plana de puntos |

Un formato de lista plana pierde qué estación registró cada punto, y con ella la
separación entre caras interior y exterior, que el flujo aprovecha.

---

## Conceptos que hay que tener claros antes de tocar el código

**Capa y pila de capas.** La capa es la unidad sobre la que opera todo: puntos,
color, escaneo de procedencia, dos condiciones independientes (visible y en
uso), y la ruta de su versión fina en disco. La pila mantiene la lista ordenada
y es el estado central de la aplicación. Vive en [`app/core/layers.py`](app/core/layers.py).

**Partir y unir.** Son las dos únicas operaciones del flujo. Recortar con una
caja, capturar con una primitiva y extraer un grupo por densidad son la misma
operación de partir con distinto criterio. Recomponer una entidad repartida
entre escaneos es unir. Que todo se reduzca a dos operaciones es lo que mantiene
el modelo pequeño y sus invariantes verificables:

- Cada punto pertenece a exactamente una capa.
- La unión de las dos mitades de una división reproduce la capa original.
- Ninguna operación destruye datos por sí sola, y lo último eliminado puede
  restaurarse.
- Todo punto conserva su escaneo de procedencia a lo largo de todo el flujo.

**Criterio de recorte, no lista de puntos.** Un recorte no guarda qué puntos
sobrevivieron, sino con qué regla se eligieron: los seis límites del prisma para
la caja, el polígono más la cámara desde la que se trazó para el lazo. Una lista
de posiciones solo tendría sentido para la nube con la que se calculó, mientras
que el criterio puede reevaluarse sobre la versión fina al exportar. De ahí que
el usuario interactúe con una nube ligera y guarde el detalle completo.

**Caché de dos resoluciones.** Un directorio por archivo con un `.ply` por
escaneo en resolución fina, del que se deriva al cargar una versión gruesa de
trabajo. La regla que gobierna la elección del voxel es que **debe elegirse en
relación con el menor espesor que se quiera resolver, no con el tamaño del
archivo**: un voxel mayor que el espesor de un elemento laminar funde sus dos
caras en una superficie intermedia que ninguna herramienta posterior puede
volver a separar.

**El espaciado medio es la escala del dato.** Es el promedio de la distancia de
cada punto a su vecino más cercano. Los parámetros que dependen de la densidad
se expresan y se conservan como proporción de ese espaciado, no como cifras
absolutas: una tolerancia en metros que funciona sobre una nube falla sobre otra
más densa. Es lo que hace el flujo trasladable a otros levantamientos.

**Las medidas que acompañan al resultado.** La *cobertura* recorre los puntos
medidos y mira a qué distancia les queda la superficie, con lo que detecta si
falta superficie donde sí había datos. La *invención* recorre la superficie y
mira a qué distancia le queda el punto medido más cercano, con lo que detecta si
hay superficie donde no había datos. Hacen falta las dos: al comparar métodos de
reconstrucción, Poisson daba una cobertura excelente de 1,75 cm mientras
fabricaba el 31 % de la superficie entregada. La *resolución* es el suelo por
debajo del cual el procedimiento no puede distinguir dos superficies, y se
reporta junto a la cobertura para que una cifra pequeña no se lea como un error
residual. La *concordancia* es el porcentaje de puntos reflejados que caen sobre
puntos reales, y es la que decide si el material de una reparación merece
confianza.

**El centro de la escena y `float32`.** Al dibujar se resta el centro de la
escena y se convierte a `float32`. Las coordenadas están en UTM, del orden de
6.300.000 m en una componente, y en esa magnitud `float32` solo distingue
valores separados por unos 0,5 m, así que dibujar sin restar el centro
colapsaría la nube a una rejilla de medio metro. En consecuencia, todo lo que se
dibuja resta ese mismo centro y todo valor que devuelve un manipulador lo suma
antes de usarse, porque los algoritmos trabajan en coordenadas del mundo.

**El código de colores es fijo.** Verde para lo que se conserva porque existe,
rojo para lo que se descarta, azul para lo que se va a generar. La tercera
categoría tiene color propio porque presentarla en verde invitaría a aceptar
material fabricado creyéndolo medido.

---

## Arquitectura

```
app/gui/        interfaz              Qt      (viewer.py añade VTK y PyVista)
app/core/       estado y entrada      Open3D  (workers.py añade Qt)
app/modules/    algoritmos puros      NumPy, SciPy, Open3D
```

Las dependencias apuntan en un solo sentido. Los módulos de `app/modules/` **no
importan Qt ni PyVista**: reciben matrices `(N, 3)` de NumPy y devuelven
matrices o máscaras booleanas, lo que permite ejercitarlos en las pruebas sin
abrir una ventana. En sentido contrario, **VTK y PyVista aparecen en un solo
archivo**, `viewer.py`: ni los paneles ni el modelo de capas saben que existen.

La interfaz puede saltarse la capa intermedia, y lo hace a menudo: cuando el
usuario mueve un control y el resultado solo hay que dibujarlo, la ventana pide
el cálculo directamente a la capa de algoritmos, porque todavía no hay nada que
registrar. La capa de datos entra en juego solo al confirmar la operación. Esas
llamadas usan importaciones diferidas, de modo que un módulo de cálculo se carga
la primera vez que se usa su herramienta y no al abrir la aplicación.

**Concurrencia.** Toda operación que supere el orden de la décima de segundo
sale del thread de la interfaz a un `QThread` de `core/workers.py`, con el mismo
patrón: el constructor recibe datos ya preparados y nunca objetos de interfaz,
`run` llama al módulo puro y captura excepciones, y el resultado sale por una
señal `finished` o `error`. Eso no acelera nada, solo evita que la ventana se
congele. Cuando el cálculo se compone de consultas independientes, como las del
refinamiento del plano de simetría, se reparte además entre núcleos. Un caso
intermedio se resuelve de otra manera: las operaciones demasiado costosas para
repetirse en cada paso de un deslizador pero demasiado breves para justificar un
thread, del orden de 300 ms, se difieren hasta que el usuario suelta el control.

**Por qué el lazo usa observadores de VTK y no una superposición de Qt.** En
Windows el widget de VTK es una ventana nativa que captura los eventos del ratón
antes de que Qt los vea. El giro de cámara con el botón derecho sí se resuelve
con un filtro de eventos de Qt, porque ahí la pulsación la recibe Qt primero.

---

## Estructura del proyecto

```
main.py                        Punto de entrada
requirements.txt

app/
├── core/
│   ├── project.py             Estado global de la aplicación
│   ├── io.py                  Despacho por formato, caché de dos resoluciones, exportación
│   ├── layers.py              LayerStack y CloudLayer: partir, unir, procedencia
│   └── workers.py             QThreads: carga, re-cacheo, DBSCAN, simetría, mallado
├── gui/
│   ├── main_window.py         Ventana, menús, orquestación de las herramientas
│   ├── viewer.py              Escena 3D, cámara, vistas fijas, manipuladores, lazo
│   ├── polygon_overlay.py     Dibujo del polígono del lazo sobre la escena
│   └── panels/
│       ├── layer_dock.py      Capas: visibilidad, uso, renombrar, unir, operaciones en bloque
│       ├── crop_dock.py       Controles de la herramienta activa, una página por herramienta
│       └── info_panel.py      Puntos del archivo y del caché, escaneos, dimensiones, espaciado
└── modules/
    ├── processing.py          Filtrado y submuestreo
    ├── segmentation.py        RANSAC propio: esfera, plano, círculo y recta, con refinamiento
    ├── primitivas.py          Las cuatro primitivas: ajuste y recortes en sus propias coordenadas
    ├── clusters.py            DBSCAN con parámetros derivados del espaciado, espaciado medio
    ├── refine_interior.py     Sub-segmentación del interior por densidad
    ├── segmentation_metrics.py  Métricas de calidad de segmentación por clase
    ├── simetria.py            Reflexión, concordancia y refinamiento del plano (Nelder-Mead)
    ├── meshing.py             Advancing Front, cierre de contornos, cobertura e invención
    └── fem.py                 Prueba de concepto suelta, ver más abajo

cgal_bridge/                   Extensión C++ con pybind11
├── src/cgal_bridge.cpp        Funciones publicadas a Python
├── CMakeLists.txt
├── COMPILACION.md             Guía de compilación paso a paso
└── *.py                       Utilidades de línea de comandos e inspección de mallas

scripts/                       Análisis fuera de la aplicación, ver más abajo
tests/                         Suite de pytest
```

### Por qué RANSAC está implementado y no tomado de una biblioteca

Se necesitaba el mismo control sobre el umbral y sobre el refinamiento posterior
en las cuatro figuras, y el de Open3D solo ajusta planos. El procedimiento tiene
tres etapas: muestreo mínimo repetido con conteo de inliers, descarte de
candidatas imposibles, y reajuste por mínimos cuadrados sobre los inliers de la
ganadora.

El umbral es lo que decide el resultado. Sobre un plano sintético con un 25 % de
outliers situados todos por encima, un umbral de 0,020 m recupera la altura
exacta, mientras que uno de 0,100 m devuelve 2,051 m, es decir, casi lo mismo
que un ajuste por mínimos cuadrados con sus 2,064 m. Un umbral mal elegido anula
la ventaja que justificaba usar RANSAC. La ventaja además solo se manifiesta
cuando los outliers se concentran a un lado de la superficie, que es lo que
ocurre en una construcción.

### El orden entre primitivas y DBSCAN no es arbitrario

La agrupación separa lo que está físicamente desconectado, y la envolvente de
una construcción es una superficie continua donde todo se toca, así que aplicada
antes de tiempo devuelve un único componente sin valor informativo. Las
primitivas retiran primero las superficies de revolución y la agrupación opera
sobre el residuo. La aplicación advierte de ese orden en el propio panel,
precisamente porque el resultado de invertirlo se confunde con un error.

### `scripts/` y `app/modules/fem.py`

`scripts/` reúne análisis que se ejecutaron fuera de la aplicación, como el
perfil de radios por bandas de altura que midió el espesor de los elementos, los
barridos de parámetros de DBSCAN y las evaluaciones de segmentación. No forman
parte del flujo de la interfaz y se dejan porque documentan cómo se calibraron
las constantes y porque sirven a quien quiera repetir esas mediciones.

`app/modules/fem.py` es una prueba de concepto temprana de un pipeline modal
(FreeCAD para la geometría, gmsh para la malla de volumen, CalculiX como solver)
que **no está conectada a la aplicación**: solo la usan `scripts/fem_toy_test.py`
y sus pruebas. Se conserva por lo que demostró sobre la viabilidad de ese
encadenamiento, pero la salida de la herramienta son mallas de superficie por
entidad, no modelos de elementos finitos. Ejecutarla requiere FreeCAD y CalculiX
instalados aparte, que no son paquetes de pip.

---

## El caso de prueba

El flujo se aplicó completo a la cara exterior del levantamiento del
Observatorio: un `.e57` de 18,5 GB con 803.234.959 puntos en 35 escaneos, sobre
una escena de 280 × 226 × 151 m.

De ahí se pasó a una nube de trabajo de 2.135.683 puntos acotada al edificio, y
en el camino el espaciado mejoró de 0,064 m a 0,020 m en lugar de empeorar,
porque la reducción proviene de descartar escena y no detalle. El resultado
fueron cinco entidades malladas y exportadas (cúpula, compuertas, base, suelo y
contrafuertes), ninguna con aristas no manifold, con superficie fabricada entre
el 0,05 % y el 0,88 %. Los contrafuertes se apartan con un 3,37 % porque se
mallaron los veinte juntos y con material residual que la limpieza no retiró, lo
que queda reflejado en la cifra.

Ninguna de las mallas es cerrada, y ese es el resultado correcto: cerrarlas
exigiría fabricar las caras que el escáner no vio.

---

## Límites conocidos

- **La frontera entre entidades adyacentes no es intrínseca.** Cada punto
  pertenece a una sola capa, así que en el encuentro entre dos elementos la
  asignación depende de la tolerancia con que se capturó cada uno. Las mallas
  vecinas arrastran una franja de incertidumbre de ese orden y no encajan entre
  sí.
- **Todo opera sobre una versión submuestreada** del levantamiento, de modo que
  las medidas de fidelidad están referidas a esa versión y no al edificio.
- **El detalle descartado al construir un caché no se recupera desde él.** La
  única vía es releer el archivo original, que es lo que hace la operación de
  re-cacheo sobre una región acotada.
- **El lazo selecciona sobre la proyección en pantalla**, así que captura a
  cualquier profundidad. Sirve para retirar elementos aislados, no para recortar
  sobre una superficie curva; para eso está la agrupación por densidad, que
  opera en tres dimensiones.
- **La reparación por simetría solo completa aquello que tenga una posición
  simétrica efectivamente medida**, y el material generado exige una revisión que
  la herramienta pide pero no puede hacer por el usuario.

---

## Licencia

Por definir.

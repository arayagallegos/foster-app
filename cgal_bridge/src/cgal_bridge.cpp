// cgal_bridge.cpp — Puente CGAL <-> Python (pybind11) para reparacion de mallas.
//
// Expone a Python lo minimo que necesita la herramienta:
//   fill_holes(V, F)  -> rellena TODOS los hoyos de la malla
//   mesh_stats(V, F)  -> diagnostico: cerrada? auto-intersecta? cuantos hoyos?
//
// Las mallas van y vuelven como arrays numpy:
//   V: (n,3) float64  vertices
//   F: (m,3) int32    triangulos (indices a V)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/boost/graph/border.h>          // extract_boundary_cycles (CGAL 6)
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>   // repair_polygon_soup
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>

#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
// duplicate_non_manifold_edges_in_polygon_soup vive aqui (no en repair_polygon_soup.h)
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/remove_outliers.h>

#include <array>
#include <iterator>

#include <cmath>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3                                          Point;
typedef CGAL::Surface_mesh<Point>                           Mesh;
typedef boost::graph_traits<Mesh>::halfedge_descriptor      halfedge_descriptor;
typedef boost::graph_traits<Mesh>::face_descriptor          face_descriptor;
typedef boost::graph_traits<Mesh>::vertex_descriptor        vertex_descriptor;


// ---------- conversion numpy -> CGAL ----------
//
// IMPORTANTE: primero se intenta construir la malla TAL CUAL (add_face). Solo si
// eso falla (hay caras no insertables, p.ej. aristas no-manifold) se recurre a
// repair/orient_polygon_soup. Pasar siempre por la "sopa" era destructivo:
// eliminaba vertices y duplicaba otros, generando hoyos que no existian en la
// malla original (BPA entraba con 8.356 bordes y salia con 24.239).
static Mesh to_mesh(py::array_t<double> V, py::array_t<int> F) {
    auto v = V.unchecked<2>();
    auto f = F.unchecked<2>();

    std::vector<Point> points;
    points.reserve(v.shape(0));
    for (py::ssize_t i = 0; i < v.shape(0); ++i)
        points.emplace_back(v(i, 0), v(i, 1), v(i, 2));

    // --- intento directo: conserva la malla exactamente como viene ---
    Mesh m;
    std::vector<vertex_descriptor> vd;
    vd.reserve(points.size());
    for (const Point& p : points) vd.push_back(m.add_vertex(p));

    bool ok = true;
    for (py::ssize_t i = 0; i < f.shape(0); ++i) {
        face_descriptor fd = m.add_face(vd[f(i, 0)], vd[f(i, 1)], vd[f(i, 2)]);
        if (fd == Mesh::null_face()) { ok = false; break; }
    }
    if (ok) return m;

    // --- respaldo: reparar como sopa de poligonos ---
    // Orden importante: primero separar explicitamente los "pellizcos"
    // (vertices/aristas no-manifold) y recien despues orientar. Si se orienta
    // sin separar, CGAL corta la malla por su cuenta y genera bordes extra.
    std::vector<std::vector<std::size_t>> polys;
    polys.reserve(f.shape(0));
    for (py::ssize_t i = 0; i < f.shape(0); ++i)
        polys.push_back({(std::size_t)f(i, 0), (std::size_t)f(i, 1),
                         (std::size_t)f(i, 2)});

    PMP::repair_polygon_soup(points, polys);
    PMP::duplicate_non_manifold_edges_in_polygon_soup(points, polys);
    PMP::orient_polygon_soup(points, polys);

    Mesh m2;
    PMP::polygon_soup_to_polygon_mesh(points, polys, m2);
    return m2;
}


// ---------- conversion CGAL -> numpy ----------
static py::tuple from_mesh(const Mesh& m) {
    std::vector<double> vs;
    std::vector<int>    fs;
    // m.num_vertices() incluye los slots liberados: es la cota correcta para
    // indexar por descriptor (number_of_vertices() seria demasiado corto).
    std::vector<int>    idx(m.num_vertices(), -1);

    int n = 0;
    for (vertex_descriptor v : m.vertices()) {
        const Point& p = m.point(v);
        vs.push_back(p.x()); vs.push_back(p.y()); vs.push_back(p.z());
        idx[(std::size_t)v] = n++;
    }
    int nf = 0;
    for (face_descriptor f : m.faces()) {
        std::vector<int> tri;
        for (vertex_descriptor v : CGAL::vertices_around_face(m.halfedge(f), m))
            tri.push_back(idx[(std::size_t)v]);
        if (tri.size() == 3) {                       // ignora caras no triangulares
            fs.insert(fs.end(), tri.begin(), tri.end());
            ++nf;
        }
    }

    py::array_t<double> V({(py::ssize_t)n, (py::ssize_t)3});
    std::memcpy(V.mutable_data(), vs.data(), vs.size() * sizeof(double));
    py::array_t<int> F({(py::ssize_t)nf, (py::ssize_t)3});
    std::memcpy(F.mutable_data(), fs.data(), (std::size_t)nf * 3 * sizeof(int));
    return py::make_tuple(V, F);
}


// ---------- API expuesta ----------

// Perimetro de un ciclo de borde (en metros) y su numero de aristas.
static std::pair<double, int> medir_hoyo(const Mesh& m, halfedge_descriptor h) {
    double per = 0.0;
    int n = 0;
    halfedge_descriptor it = h;
    do {
        per += std::sqrt(CGAL::squared_distance(m.point(source(it, m)),
                                                m.point(target(it, m))));
        ++n;
        it = next(it, m);
    } while (it != h);
    return {per, n};
}


// Rellena hoyos. `fair` suaviza el parche (requiere Eigen).
// `max_perimetro`: si es > 0, SOLO se rellenan los hoyos cuyo perimetro sea
// menor a ese valor (en metros). Los mayores se dejan abiertos, porque
// corresponden a zonas sin datos y taparlos seria fabricar geometria.
static py::tuple fill_holes(py::array_t<double> V, py::array_t<int> F,
                            bool fair = true, double density = 2.0,
                            double max_perimetro = 0.0) {
    Mesh m = to_mesh(V, F);

    std::vector<halfedge_descriptor> borders;
    CGAL::extract_boundary_cycles(m, std::back_inserter(borders));

    // No se recogen los iteradores de salida (no se usan): asi el codigo sirve
    // igual en CGAL 5.x y 6.x, que cambiaron esa parte de la API.
    for (halfedge_descriptor h : borders) {
        if (max_perimetro > 0.0 && medir_hoyo(m, h).first > max_perimetro)
            continue;                       // hoyo grande -> se deja abierto
#ifdef CGAL_EIGEN3_ENABLED
        if (fair) {
            PMP::triangulate_refine_and_fair_hole(
                m, h, CGAL::parameters::density_control_factor(density));
            continue;
        }
#endif
        (void)fair;
        PMP::triangulate_and_refine_hole(
            m, h, CGAL::parameters::density_control_factor(density));
    }
    return from_mesh(m);
}


// ---------- reconstruccion de superficie ----------

// Uniformiza la densidad: deja UN punto por celda de una rejilla de lado
// `cell_size`. Es el paso que recomienda CGAL cuando el muestreo es irregular
// (nubes de escaner, o mezcladas con puntos sinteticos de otra densidad).
static py::array_t<double> grid_simplify(py::array_t<double> P, double cell_size) {
    auto p = P.unchecked<2>();
    std::vector<Point> pts;
    pts.reserve(p.shape(0));
    for (py::ssize_t i = 0; i < p.shape(0); ++i)
        pts.emplace_back(p(i, 0), p(i, 1), p(i, 2));

    pts.erase(CGAL::grid_simplify_point_set(pts, cell_size), pts.end());

    py::array_t<double> out({(py::ssize_t)pts.size(), (py::ssize_t)3});
    auto o = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        o(i, 0) = pts[i].x(); o(i, 1) = pts[i].y(); o(i, 2) = pts[i].z();
    }
    return out;
}


// Quita outliers: descarta el `percent`% de puntos mas alejados de sus k vecinos.
static py::array_t<double> remove_outliers(py::array_t<double> P, int k,
                                           double percent) {
    auto p = P.unchecked<2>();
    std::vector<Point> pts;
    pts.reserve(p.shape(0));
    for (py::ssize_t i = 0; i < p.shape(0); ++i)
        pts.emplace_back(p(i, 0), p(i, 1), p(i, 2));

    pts.erase(CGAL::remove_outliers<CGAL::Sequential_tag>(
                  pts, k, CGAL::parameters::threshold_percent(percent)),
              pts.end());

    py::array_t<double> out({(py::ssize_t)pts.size(), (py::ssize_t)3});
    auto o = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        o(i, 0) = pts[i].x(); o(i, 1) = pts[i].y(); o(i, 2) = pts[i].z();
    }
    return out;
}


// Prioridad con LIMITE DE PERIMETRO: rechaza (prioridad infinita) los triangulos
// cuyo perimetro supere `bound`. Es el mecanismo que ofrece CGAL para evitar que
// el frente "puentee" vacios con triangulos estirados — que es justo lo que
// genera superficie inventada entre islas de puntos.
struct PerimetroAcotado {
    double bound;
    explicit PerimetroAcotado(double b) : bound(b) {}

    template <typename AdvancingFront, typename Cell_handle>
    double operator()(const AdvancingFront& adv, Cell_handle& c,
                      const int& index) const {
        if (bound == 0)
            return adv.smallest_radius_delaunay_sphere(c, index);
        const Point& a = c->vertex((index + 1) % 4)->point();
        const Point& b = c->vertex((index + 2) % 4)->point();
        const Point& d = c->vertex((index + 3) % 4)->point();
        double per = std::sqrt(CGAL::squared_distance(a, b))
                   + std::sqrt(CGAL::squared_distance(b, d))
                   + std::sqrt(CGAL::squared_distance(a, d));
        if (per > bound) return adv.infinity();          // triangulo demasiado grande
        return adv.smallest_radius_delaunay_sphere(c, index);
    }
};


// Advancing Front: reconstruye interpolando los puntos REALES (como Ball
// Pivoting) pero produciendo una malla MANIFOLD y orientada por construccion,
// admitiendo bordes (superficies abiertas). No necesita normales.
//   radius_ratio_bound: deteccion de bordes/hoyos (distingue zona submuestreada
//                       de borde real)
//   beta: umbral de angulo diedro para la "zona de plausibilidad"
//   max_perimetro: si >0, descarta triangulos de perimetro mayor (evita puentear
//                  vacios). 0 = sin limite.
static py::tuple advancing_front(py::array_t<double> P,
                                 double radius_ratio_bound = 5.0,
                                 double beta = 0.52,
                                 double max_perimetro = 0.0) {
    auto p = P.unchecked<2>();
    std::vector<Point> pts;
    pts.reserve(p.shape(0));
    for (py::ssize_t i = 0; i < p.shape(0); ++i)
        pts.emplace_back(p(i, 0), p(i, 1), p(i, 2));

    typedef std::array<std::size_t, 3> Facet;
    std::vector<Facet> facets;
    PerimetroAcotado prio(max_perimetro);
    CGAL::advancing_front_surface_reconstruction(
        pts.begin(), pts.end(), std::back_inserter(facets), prio,
        radius_ratio_bound, beta);

    py::array_t<double> V({(py::ssize_t)pts.size(), (py::ssize_t)3});
    auto v = V.mutable_unchecked<2>();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        v(i, 0) = pts[i].x(); v(i, 1) = pts[i].y(); v(i, 2) = pts[i].z();
    }
    py::array_t<int> F({(py::ssize_t)facets.size(), (py::ssize_t)3});
    auto f = F.mutable_unchecked<2>();
    for (std::size_t i = 0; i < facets.size(); ++i)
        for (int j = 0; j < 3; ++j) f(i, j) = (int)facets[i][j];
    return py::make_tuple(V, F);
}


// Cose aristas de borde que ya coinciden geometricamente pero quedaron
// duplicadas (tipico de Ball Pivoting, que triangula cada parche por separado).
// NO inventa geometria: solo une lo que ya estaba pegado. Es el paso previo
// natural antes de rellenar hoyos.
static py::tuple stitch(py::array_t<double> V, py::array_t<int> F) {
    Mesh m = to_mesh(V, F);
    PMP::stitch_borders(m);
    return from_mesh(m);
}


// Distribucion de tamanos de hoyo: para elegir el umbral con criterio.
static py::list hole_sizes(py::array_t<double> V, py::array_t<int> F) {
    Mesh m = to_mesh(V, F);
    std::vector<halfedge_descriptor> borders;
    CGAL::extract_boundary_cycles(m, std::back_inserter(borders));
    py::list out;
    for (halfedge_descriptor h : borders) {
        auto pn = medir_hoyo(m, h);
        py::dict d;
        d["perimetro"] = pn.first;
        d["aristas"]   = pn.second;
        out.append(d);
    }
    return out;
}


// Diagnostico de calidad: lo que hay que verificar al exportar.
static py::dict mesh_stats(py::array_t<double> V, py::array_t<int> F) {
    Mesh m = to_mesh(V, F);

    std::vector<halfedge_descriptor> borders;
    CGAL::extract_boundary_cycles(m, std::back_inserter(borders));

    py::dict d;
    d["n_vertices"]      = (int)m.number_of_vertices();
    d["n_faces"]         = (int)m.number_of_faces();
    d["n_holes"]         = (int)borders.size();
    d["is_closed"]       = CGAL::is_closed(m);
    d["self_intersects"] = PMP::does_self_intersect(m);
    d["is_outward"]      = CGAL::is_closed(m) ? PMP::is_outward_oriented(m) : false;
    return d;
}


PYBIND11_MODULE(cgal_bridge, mod) {
    mod.doc() = "Puente CGAL para reparacion de mallas (Scan-to-mesh)";

    mod.def("fill_holes", &fill_holes,
            py::arg("vertices"), py::arg("faces"),
            py::arg("fair") = true, py::arg("density") = 2.0,
            py::arg("max_perimetro") = 0.0,
            "Rellena hoyos. max_perimetro>0 -> solo los de perimetro menor a ese "
            "valor (m); los mayores se dejan abiertos. Devuelve (V, F).");

    mod.def("grid_simplify", &grid_simplify,
            py::arg("points"), py::arg("cell_size"),
            "Uniformiza la densidad: un punto por celda de lado cell_size (m).");

    mod.def("remove_outliers", &remove_outliers,
            py::arg("points"), py::arg("k") = 24, py::arg("percent") = 5.0,
            "Quita el percent%% de puntos mas alejados de sus k vecinos.");

    mod.def("advancing_front", &advancing_front,
            py::arg("points"), py::arg("radius_ratio_bound") = 5.0,
            py::arg("beta") = 0.52, py::arg("max_perimetro") = 0.0,
            "Reconstruccion Advancing Front: interpola los puntos reales y produce "
            "malla MANIFOLD orientada con bordes. max_perimetro>0 descarta "
            "triangulos grandes (evita puentear vacios). Devuelve (V, F).");

    mod.def("stitch", &stitch,
            py::arg("vertices"), py::arg("faces"),
            "Cose aristas de borde duplicadas pero coincidentes (PMP::stitch_borders). "
            "No inventa geometria. Devuelve (V, F).");

    mod.def("hole_sizes", &hole_sizes,
            py::arg("vertices"), py::arg("faces"),
            "Lista de {perimetro, aristas} por hoyo, para elegir el umbral.");

    mod.def("mesh_stats", &mesh_stats,
            py::arg("vertices"), py::arg("faces"),
            "Diagnostico: n_holes, is_closed, self_intersects, is_outward.");
}

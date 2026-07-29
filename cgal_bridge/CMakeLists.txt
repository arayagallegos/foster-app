cmake_minimum_required(VERSION 3.20)
project(cgal_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Python + pybind11 -------------------------------------------------
# pybind11 se toma del venv del proyecto (pip install pybind11), asi no hay
# que instalarlo por separado.
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# --- CGAL --------------------------------------------------------------
find_package(CGAL REQUIRED)

# Eigen habilita el "fairing" (suavizado del parche del hoyo). Es opcional:
# sin el, igual se cierran los hoyos, pero el parche queda menos liso.
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
  message(STATUS "Eigen3 encontrado: fairing HABILITADO")
else()
  message(WARNING "Eigen3 NO encontrado: se cerraran los hoyos sin fairing")
endif()

# --- Modulo ------------------------------------------------------------
pybind11_add_module(cgal_bridge src/cgal_bridge.cpp)
target_link_libraries(cgal_bridge PRIVATE CGAL::CGAL)

if(Eigen3_FOUND)
  target_link_libraries(cgal_bridge PRIVATE Eigen3::Eigen)
  target_compile_definitions(cgal_bridge PRIVATE CGAL_EIGEN3_ENABLED)
endif()

if(MSVC)
  # CGAL usa plantillas muy pesadas: sin /bigobj el enlace falla en Windows.
  target_compile_options(cgal_bridge PRIVATE /bigobj /EHsc)
endif()

# Deja el .pyd al lado del CMakeLists para que Python lo encuentre facil
set_target_properties(cgal_bridge PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY         "${CMAKE_CURRENT_SOURCE_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_CURRENT_SOURCE_DIR}"
)

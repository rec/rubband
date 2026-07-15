cmake_minimum_required(VERSION 3.20)

project(rubband LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(RUBBERBAND QUIET IMPORTED_TARGET rubberband)
endif()

if(RUBBERBAND_FOUND)
    set(RUBBERBAND_TARGET PkgConfig::RUBBERBAND)
else()
    find_path(RUBBERBAND_INCLUDE_DIR rubberband/RubberBandStretcher.h)
    find_library(RUBBERBAND_LIBRARY NAMES rubberband RubberBand)

    if(NOT RUBBERBAND_INCLUDE_DIR OR NOT RUBBERBAND_LIBRARY)
        message(
            FATAL_ERROR
            "Could not find Rubber Band. Install librubberband and make it "
            "discoverable with pkg-config, CMAKE_PREFIX_PATH, or vcpkg."
        )
    endif()

    add_library(rubberband::rubberband UNKNOWN IMPORTED)
    set_target_properties(
        rubberband::rubberband
        PROPERTIES
            IMPORTED_LOCATION "${RUBBERBAND_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${RUBBERBAND_INCLUDE_DIR}"
    )
    set(RUBBERBAND_TARGET rubberband::rubberband)
    set(RUBBERBAND_INCLUDE_DIRS "${RUBBERBAND_INCLUDE_DIR}")
endif()

include(CheckCXXSourceCompiles)

set(_RUBBAND_PREVIOUS_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_INCLUDES "${RUBBERBAND_INCLUDE_DIRS}")
check_cxx_source_compiles(
    "
    #include <rubberband/RubberBandStretcher.h>
    #ifndef RUBBERBAND_API_MAJOR_VERSION
    #error Rubber Band API version macros are unavailable.
    #endif
    #if RUBBERBAND_API_MAJOR_VERSION < 3
    #error Rubber Band API 3.0 or newer is required.
    #endif
    int main() { return 0; }
    "
    RUBBAND_HAS_RUBBERBAND_API_3
)
set(CMAKE_REQUIRED_INCLUDES "${_RUBBAND_PREVIOUS_REQUIRED_INCLUDES}")

if(NOT RUBBAND_HAS_RUBBERBAND_API_3)
    message(
        FATAL_ERROR
        "Rubband requires Rubber Band API 3.0 or newer. Install Rubber Band "
        "4.x and make it discoverable with pkg-config, CMAKE_PREFIX_PATH, or vcpkg."
    )
endif()

nanobind_add_module(_rubband rubband/_rubband.cpp)
target_compile_features(_rubband PRIVATE cxx_std_17)
target_link_libraries(_rubband PRIVATE "${RUBBERBAND_TARGET}")

install(TARGETS _rubband LIBRARY DESTINATION rubband)

cmake_minimum_required(VERSION 3.20)

project(rubband LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(RUBBERBAND REQUIRED IMPORTED_TARGET rubberband)

nanobind_add_module(_rubband src/rubband/_rubband.cpp)
target_compile_features(_rubband PRIVATE cxx_std_17)
target_link_libraries(_rubband PRIVATE PkgConfig::RUBBERBAND)

install(TARGETS _rubband LIBRARY DESTINATION rubband)

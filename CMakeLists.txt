cmake_minimum_required(VERSION 3.13)
project(ci_switch C)

set(CMAKE_C_STANDARD 11)
find_package(PkgConfig REQUIRED)

pkg_check_modules(DPU REQUIRED IMPORTED_TARGET dpu)

add_executable(ci-switch main.c)

target_include_directories(ci-switch PRIVATE PkgConfig::DPU)
target_link_libraries(ci-switch PUBLIC PkgConfig::DPU)

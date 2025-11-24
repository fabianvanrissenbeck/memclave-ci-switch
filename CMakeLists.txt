cmake_minimum_required(VERSION 3.13)
project(ci_switch C)

set(CMAKE_C_STANDARD 11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(DPU REQUIRED IMPORTED_TARGET dpu)

find_program(LD ld REQUIRED)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fsl.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${LD}
    ARGS
        -r -b binary -o ${CMAKE_CURRENT_BINARY_DIR}/fsl.o fsl
    VERBATIM
    MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fsl
)

add_custom_target(fsl DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fsl.o)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/fsl.o PROPERTIES GENERATED TRUE)

add_executable(ci-switch main.c)
add_dependencies(ci-switch fsl)

add_subdirectory(common)
target_link_libraries(ci-switch PRIVATE common ${CMAKE_CURRENT_BINARY_DIR}/fsl.o)

target_include_directories(ci-switch PRIVATE PkgConfig::DPU)
target_link_libraries(ci-switch PUBLIC PkgConfig::DPU)

cmake_minimum_required(VERSION 3.13)
project(ci_switch C)
include(ExternalProject)

set(CMAKE_C_STANDARD 11)
set(UPMEM_TOOLCHAIN_FILE "/usr/share/upmem/cmake/dpu.cmake")

ExternalProject_Add(
        fsl
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ime
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -S <SOURCE_DIR>
            -B <BINARY_DIR>
            -DCMAKE_TOOLCHAIN_FILE=${UPMEM_TOOLCHAIN_FILE}
        BUILD_COMMAND ${CMAKE_COMMAND}
            --build <BINARY_DIR>
            --target fsl
        INSTALL_COMMAND ""
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(DPU REQUIRED IMPORTED_TARGET dpu)

find_program(LD ld REQUIRED)

ExternalProject_Get_Property(fsl BINARY_DIR)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fsl.o
    WORKING_DIRECTORY ${BINARY_DIR}/fsl # necessary so that the symbol is called _binary_fsl_start :)
    COMMAND ${LD}
    ARGS
        -r -b binary -o ${CMAKE_CURRENT_BINARY_DIR}/fsl.o fsl
    VERBATIM
    DEPENDS fsl
)

add_custom_target(fsl-bin DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fsl.o)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/fsl.o PROPERTIES GENERATED TRUE)
add_dependencies(fsl-bin fsl)

add_executable(ci-switch main.c)
add_dependencies(ci-switch fsl-bin)

add_subdirectory(common)
target_link_libraries(ci-switch PRIVATE common ${CMAKE_CURRENT_BINARY_DIR}/fsl.o)

target_include_directories(ci-switch PRIVATE PkgConfig::DPU)
target_link_libraries(ci-switch PUBLIC PkgConfig::DPU)

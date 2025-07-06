cmake_minimum_required(VERSION 3.13)
project(ci_switch C)

set(CMAKE_C_STANDARD 11)

find_package(PkgConfig REQUIRED)
find_program(DPU-Clang NAMES dpu-upmem-dpurte-clang PATHS /opt/upmem/upmem-2025.1.0-Linux-x86_64/bin NO_DEFAULT_PATH REQUIRED)

pkg_check_modules(DPU REQUIRED IMPORTED_TARGET dpu)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/reset.elf
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND ${DPU-Clang}
        VERBATIM
        ARGS
            -nostartfiles
            -o ${CMAKE_CURRENT_BINARY_DIR}/reset.elf
            ${CMAKE_CURRENT_SOURCE_DIR}/misc/reset.S
        MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/misc/reset.S
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/misc/reset.S
)

add_custom_target(dpu-reset-kernel DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/reset.elf)

add_executable(ci-switch main.c)
add_dependencies(ci-switch dpu-reset-kernel)

add_subdirectory(common)
target_link_libraries(ci-switch PRIVATE common)

target_include_directories(ci-switch PRIVATE PkgConfig::DPU)
target_link_libraries(ci-switch PUBLIC PkgConfig::DPU)

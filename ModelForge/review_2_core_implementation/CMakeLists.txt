cmake_minimum_required(VERSION 3.16)

project(ModelForgeCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The terminal studio prints Unicode box-drawing glyphs from UTF-8 sources.
if(MSVC)
    add_compile_options(/utf-8)
endif()

option(MODELFORGE_WITH_ONNX "Enable the optional ONNX/Protobuf loader" OFF)
option(MODELFORGE_WITH_ONNX_RUNTIME "Enable ONNX Runtime equivalence verification" OFF)

if(MODELFORGE_WITH_ONNX_RUNTIME AND NOT MODELFORGE_WITH_ONNX)
    message(FATAL_ERROR "MODELFORGE_WITH_ONNX_RUNTIME also requires MODELFORGE_WITH_ONNX=ON")
endif()

add_library(modelforge_core
    src/adaptive.cpp
    src/analysis.cpp
    src/codegen.cpp
    src/diagnostics.cpp
    src/guardian.cpp
    src/ir.cpp
    src/loader.cpp
    src/model.cpp
    src/optimizer.cpp
    src/research.cpp
    src/shrink.cpp
    src/validator.cpp
    src/verifier.cpp
)

target_include_directories(modelforge_core PUBLIC src)

if(MODELFORGE_WITH_ONNX)
    find_package(Protobuf REQUIRED)
    find_path(MODELFORGE_ONNX_INCLUDE_DIR onnx/onnx_pb.h)
    find_library(MODELFORGE_ONNX_LIBRARY NAMES onnx)

    if(NOT MODELFORGE_ONNX_INCLUDE_DIR OR NOT MODELFORGE_ONNX_LIBRARY)
        message(FATAL_ERROR "MODELFORGE_WITH_ONNX requires ONNX headers and the ONNX library")
    endif()

    target_compile_definitions(modelforge_core PUBLIC MODELFORGE_WITH_ONNX=1)
    target_include_directories(modelforge_core PRIVATE ${MODELFORGE_ONNX_INCLUDE_DIR})
    target_link_libraries(modelforge_core PRIVATE protobuf::libprotobuf ${MODELFORGE_ONNX_LIBRARY})
endif()

if(MODELFORGE_WITH_ONNX_RUNTIME)
    find_path(MODELFORGE_ONNX_RUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
    find_library(MODELFORGE_ONNX_RUNTIME_LIBRARY NAMES onnxruntime)

    if(NOT MODELFORGE_ONNX_RUNTIME_INCLUDE_DIR OR NOT MODELFORGE_ONNX_RUNTIME_LIBRARY)
        message(FATAL_ERROR "MODELFORGE_WITH_ONNX_RUNTIME requires ONNX Runtime headers and library")
    endif()

    target_compile_definitions(modelforge_core PUBLIC MODELFORGE_WITH_ONNX_RUNTIME=1)
    target_include_directories(modelforge_core PRIVATE ${MODELFORGE_ONNX_RUNTIME_INCLUDE_DIR})
    target_link_libraries(modelforge_core PRIVATE ${MODELFORGE_ONNX_RUNTIME_LIBRARY})
endif()

add_executable(modelforge src/main.cpp)
target_link_libraries(modelforge PRIVATE modelforge_core)

add_executable(modelforge_tests tests/test_runner.cpp)
target_link_libraries(modelforge_tests PRIVATE modelforge_core)

add_executable(modelforge_guardian_eval tests/guardian_eval.cpp)
target_link_libraries(modelforge_guardian_eval PRIVATE modelforge_core)

# Research benchmark and the terminal studio. The dashboard lives at the
# repository root so it can be opened straight from VS Code's explorer.
get_filename_component(MODELFORGE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(modelforge_bench tools/bench.cpp)
add_executable(modelforge_studio tools/studio.cpp)
foreach(tool modelforge_bench modelforge_studio)
    target_link_libraries(${tool} PRIVATE modelforge_core)
    target_include_directories(${tool} PRIVATE tools)
    target_compile_definitions(${tool} PRIVATE
        MODELFORGE_DASHBOARD_DIR="${MODELFORGE_REPO_ROOT}/dashboard"
        MODELFORGE_DEFAULT_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/models/iris_demo.mforge"
        MODELFORGE_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")
endforeach()

# Keep multi-configuration generators (such as Visual Studio) predictable
# when this project is built from the workspace wrapper or directly.
set(MODELFORGE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")
set_target_properties(modelforge modelforge_tests modelforge_guardian_eval
                      modelforge_bench modelforge_studio PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${MODELFORGE_OUTPUT_DIR}"
)
set_target_properties(modelforge_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${MODELFORGE_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${MODELFORGE_OUTPUT_DIR}"
)

enable_testing()
add_test(
    NAME modelforge_tests
    COMMAND modelforge_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures
)
add_test(
    NAME modelforge_guardian_eval
    COMMAND modelforge_guardian_eval ${CMAKE_CURRENT_SOURCE_DIR}/models/iris_demo.mforge
)
add_test(
    NAME modelforge_bench_quick
    COMMAND modelforge_bench --quick --no-color --ascii
            --out ${CMAKE_BINARY_DIR}/research_quick
            --dashboard ${CMAKE_BINARY_DIR}/research_quick/dashboard
)
add_test(
    NAME modelforge_studio_demo
    COMMAND modelforge_studio --demo --no-color --ascii
            --dashboard ${CMAKE_BINARY_DIR}/studio_dashboard
)
set_tests_properties(modelforge_studio_demo PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

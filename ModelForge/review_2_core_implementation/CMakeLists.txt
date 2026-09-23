cmake_minimum_required(VERSION 3.16)

project(ModelForgeCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MODELFORGE_WITH_ONNX "Enable the optional ONNX/Protobuf loader" OFF)
option(MODELFORGE_WITH_ONNX_RUNTIME "Enable ONNX Runtime equivalence verification" OFF)

if(MODELFORGE_WITH_ONNX_RUNTIME AND NOT MODELFORGE_WITH_ONNX)
    message(FATAL_ERROR "MODELFORGE_WITH_ONNX_RUNTIME also requires MODELFORGE_WITH_ONNX=ON")
endif()

add_library(modelforge_core
    src/codegen.cpp
    src/diagnostics.cpp
    src/guardian.cpp
    src/ir.cpp
    src/loader.cpp
    src/model.cpp
    src/optimizer.cpp
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

# Keep multi-configuration generators (such as Visual Studio) predictable
# when this project is built from the workspace wrapper or directly.
set(MODELFORGE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")
set_target_properties(modelforge modelforge_tests PROPERTIES
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

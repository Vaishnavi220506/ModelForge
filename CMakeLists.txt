cmake_minimum_required(VERSION 3.16)

project(ModelForgeWorkspace LANGUAGES CXX)

# The VS Code workspace is opened at Compiler/, while the implementation lives
# in ModelForge/review_2_core_implementation/. This wrapper keeps both layouts
# working with the standard `cmake -S . -B build` command.
enable_testing()

add_subdirectory(
    ModelForge/review_2_core_implementation
    ModelForgeReview2
)

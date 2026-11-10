cmake_minimum_required(VERSION 3.22)
project(kneejerk_jni LANGUAGES C CXX)

# KNEEJERK_ROOT: the kneejerk checkout (its CMakeLists builds libsystemone and llama.cpp from the submodule).
if (NOT DEFINED KNEEJERK_ROOT)
    get_filename_component(KNEEJERK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
endif()

# One shared library for the app: everything else static inside it.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SYSTEMONE_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)      # cross-compiling: no -march=native, the arch comes from build.gradle
if (NOT GGML_CPU_ARM_ARCH)
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif()
# Gradle configures the native build as Debug for a debug app; llama.cpp at -O0 takes seconds per decision
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(${KNEEJERK_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/kneejerk)

add_library(kneejerk_jni SHARED jni.cpp)
target_link_libraries(kneejerk_jni PRIVATE systemone log android)
target_compile_options(kneejerk_jni PRIVATE -fvisibility=hidden)

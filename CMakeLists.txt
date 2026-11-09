cmake_minimum_required(VERSION 3.21)
project(systemone LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp: the submodule, or a checkout given by SYSTEMONE_LLAMA_CPP_DIR. The decision code is
# written against release b12207; a newer release needs src/upstream/ refreshed from its tools/server.
set(SYSTEMONE_LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "llama.cpp source tree")
option(SYSTEMONE_BUILD_CLI "build systemone-cli" ON)

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)   # mtmd lives under tools/
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
if (NOT EXISTS "${SYSTEMONE_LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp is not checked out at ${SYSTEMONE_LLAMA_CPP_DIR}.\n"
        "Run:  git submodule update --init --depth 1\n"
        "or point SYSTEMONE_LLAMA_CPP_DIR at a llama.cpp checkout (release b12207).")
endif()
add_subdirectory(${SYSTEMONE_LLAMA_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)

add_library(systemone
    src/systemone.cpp
    src/upstream/server-decision.cpp
)
target_include_directories(systemone
    PUBLIC  include
    PRIVATE src/shim src/upstream
            ${SYSTEMONE_LLAMA_CPP_DIR}/include
            ${SYSTEMONE_LLAMA_CPP_DIR}/ggml/include
            ${SYSTEMONE_LLAMA_CPP_DIR}/common
            ${SYSTEMONE_LLAMA_CPP_DIR}/vendor
            ${SYSTEMONE_LLAMA_CPP_DIR}/src            # llama-ext.h, llama-arch.h (staging API)
            ${SYSTEMONE_LLAMA_CPP_DIR}/tools/mtmd
)
target_link_libraries(systemone PUBLIC llama PRIVATE llama-common mtmd)
set_target_properties(systemone PROPERTIES POSITION_INDEPENDENT_CODE ON)

# iOS 12 and older: libc++ has no std::filesystem. ggml links it for the dynamic backend loader, which
# llama_backend_init() never runs when a backend is built in. Weak-link libc++, so these missing symbols do not stop the launch.
if (CMAKE_SYSTEM_NAME STREQUAL "iOS" AND CMAKE_OSX_DEPLOYMENT_TARGET AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 13.0)
    target_link_options(systemone INTERFACE -nostdlib++ "LINKER:-weak-lc++")
endif()

if (SYSTEMONE_BUILD_CLI)
    add_executable(systemone-cli tools/cli.cpp)
    target_link_libraries(systemone-cli PRIVATE systemone)
endif()

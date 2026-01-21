# CMake version and project name
cmake_minimum_required(VERSION 3.10)
project(TinyML VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find xsimd library
find_package(xsimd QUIET)

if(NOT xsimd_FOUND)
    # Fetch xsimd if not found
    include(FetchContent)
    FetchContent_Declare(
        xsimd
        URL https://github.com/xtensor-stack/xsimd/archive/refs/tags/12.1.1.tar.gz
    )
    FetchContent_MakeAvailable(xsimd)
endif()

# Real-time optimizations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Set output directories for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Collect all source files from the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/SIMDOperations.cpp)
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/NEONOperations.cpp)
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/RealTimeTransformer.cpp)

# Print out the source files collected
message(STATUS "Source files: ${SOURCES}")

# Create the main library target
add_library(TinyML ${SOURCES})

# Ensure that the include directories for the library are available to targets that link with the library
target_include_directories(TinyML PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Link with math library if needed
if(UNIX AND NOT APPLE)
    target_link_libraries(TinyML m)
endif()

# Link with xsimd if available
if(xsimd_FOUND)
    target_link_libraries(TinyML xsimd::xsimd)
    target_include_directories(TinyML PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    # Try to find xsimd in the build directory
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(TinyML PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()

# Print the include directories being set
message(STATUS "Include directory for TinyML: ${PROJECT_SOURCE_DIR}/include")

# GoogleTest integration (optional)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Tests
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")

# Print out the test source files collected
message(STATUS "Test source files: ${TEST_SOURCES}")

# Create test executable
add_executable(TinyMLTests ${TEST_SOURCES})

# Link the test executable with Google Test and your library
target_link_libraries(TinyMLTests TinyML gtest gtest_main)

# Ensure the test target also gets the correct include directories
target_include_directories(TinyMLTests PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(TinyMLTests xsimd::xsimd)
    target_include_directories(TinyMLTests PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(TinyMLTests PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()

# Print the include directories being set for the test target
message(STATUS "Include directory for TinyMLTests: ${PROJECT_SOURCE_DIR}/include")

# Basic Test executable
add_executable(BasicTests tests/basic_tests.cpp)
target_link_libraries(BasicTests TinyML gtest gtest_main)
target_include_directories(BasicTests PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(BasicTests xsimd::xsimd)
    target_include_directories(BasicTests PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(BasicTests PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME BasicTests COMMAND BasicTests)

# Phase 2 Attention Test executable
add_executable(Phase2AttentionTest tests/test_phase2_attention_xsimd.cpp)
target_link_libraries(Phase2AttentionTest TinyML gtest gtest_main)
target_include_directories(Phase2AttentionTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase2AttentionTest xsimd::xsimd)
    target_include_directories(Phase2AttentionTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase2AttentionTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
# Phase 2 Simple Test executable
add_executable(Phase2SimpleTest tests/test_phase2_simple.cpp)
target_link_libraries(Phase2SimpleTest TinyML gtest gtest_main)
target_include_directories(Phase2SimpleTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase2SimpleTest xsimd::xsimd)
    target_include_directories(Phase2SimpleTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase2SimpleTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
# Phase 3 Dynamic Test executable
add_executable(Phase3DynamicTest tests/test_phase3_dynamic_new.cpp)
target_link_libraries(Phase3DynamicTest TinyML gtest gtest_main)
target_include_directories(Phase3DynamicTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase3DynamicTest xsimd::xsimd)
    target_include_directories(Phase3DynamicTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase3DynamicTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
# Phase 4 Transformer Test executable
add_executable(Phase4TransformerTest tests/test_phase4_transformer_new.cpp)
target_link_libraries(Phase4TransformerTest TinyML gtest gtest_main)
target_include_directories(Phase4TransformerTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase4TransformerTest xsimd::xsimd)
    target_include_directories(Phase4TransformerTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase4TransformerTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
# Phase 4 Simple Transformer Test executable
add_executable(Phase4SimpleTest tests/test_phase4_simple.cpp)
target_link_libraries(Phase4SimpleTest TinyML gtest gtest_main)
target_include_directories(Phase4SimpleTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase4SimpleTest xsimd::xsimd)
    target_include_directories(Phase4SimpleTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase4SimpleTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
# Phase 4 Fixed Transformer Test executable
add_executable(Phase4FixedTest tests/test_phase4_fixed.cpp)
target_link_libraries(Phase4FixedTest TinyML gtest gtest_main)
target_include_directories(Phase4FixedTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase4FixedTest xsimd::xsimd)
    target_include_directories(Phase4FixedTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase4FixedTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME Phase4FixedTest COMMAND Phase4FixedTest)

# SIMD Benchmark executable (using XSIMD)
add_executable(SIMDBenchmark benchmarks/benchmark_xsimd.cpp)
target_link_libraries(SIMDBenchmark TinyML gtest gtest_main)
target_include_directories(SIMDBenchmark PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(SIMDBenchmark xsimd::xsimd)
    target_include_directories(SIMDBenchmark PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(SIMDBenchmark PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME SIMDBenchmark COMMAND SIMDBenchmark)

# Lightweight Attention Test executable
add_executable(LightweightAttentionTest tests/test_lightweight_attention.cpp)
target_link_libraries(LightweightAttentionTest TinyML gtest gtest_main)
target_include_directories(LightweightAttentionTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
add_test(NAME LightweightAttentionTest COMMAND LightweightAttentionTest)

# Attention Benchmark executable
add_executable(AttentionBenchmark benchmarks/benchmark_attention.cpp)
target_link_libraries(AttentionBenchmark TinyML gtest gtest_main)
target_include_directories(AttentionBenchmark PUBLIC ${PROJECT_SOURCE_DIR}/include)
add_test(NAME AttentionBenchmark COMMAND AttentionBenchmark)

# Simple Attention Benchmark executable (standalone)
add_executable(SimpleAttentionBenchmark benchmarks/benchmark_simple_attention.cpp)
target_include_directories(SimpleAttentionBenchmark PUBLIC ${PROJECT_SOURCE_DIR}/include)
add_test(NAME SimpleAttentionBenchmark COMMAND SimpleAttentionBenchmark)

# Debugging: Print the include directories that will be passed to the compiler
get_target_property(INCLUDE_DIRS TinyML INCLUDE_DIRECTORIES)
message(STATUS "TinyML include directories: ${INCLUDE_DIRS}")

get_target_property(INCLUDE_DIRS_TEST TinyMLTests INCLUDE_DIRECTORIES)
message(STATUS "TinyMLTests include directories: ${INCLUDE_DIRS_TEST}")

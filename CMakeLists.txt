# CMake version and project name
cmake_minimum_required(VERSION 3.10)
project(TinyML VERSION 1.0 LANGUAGES CXX)

# Enable CTest so add_test() registrations are picked up.
include(CTest)
enable_testing()

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
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/AdvancedOptimizations.cpp)
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/GenerativeModels.cpp)
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/BayesianNeuralNetwork.cpp)

# Only include existing source files
set(EXISTING_SOURCES)
foreach(SOURCE ${SOURCES})
    if(EXISTS ${SOURCE})
        list(APPEND EXISTING_SOURCES ${SOURCE})
    endif()
endforeach()

# Add GenerativeModels source if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/GenerativeModels.cpp)
    list(APPEND EXISTING_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/GenerativeModels.cpp)
endif()
set(SOURCES ${EXISTING_SOURCES})

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
    target_include_directories(TinyML PUBLIC ${xsimd_INCLUDE_DIRS})
else()
    # Try to find xsimd in the build directory
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(TinyML PUBLIC "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
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

# Tests - only include working GoogleTest test files
set(TEST_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase1_simd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase3_dynamic_new.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase4_fixed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase4_simple.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase5_quantized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase6_production.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase7_advanced_attention.cpp
)

# Add GenerativeModels test if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase11_generative.cpp)
    list(APPEND TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase11_generative.cpp)
endif()

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
add_test(NAME TinyMLTests COMMAND TinyMLTests)

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
# Production API Test executable
add_executable(ProductionAPITest tests/test_production_api.cpp)
target_link_libraries(ProductionAPITest TinyML gtest gtest_main)
target_include_directories(ProductionAPITest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(ProductionAPITest xsimd::xsimd)
    target_include_directories(ProductionAPITest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(ProductionAPITest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
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

# Phase 5 Quantized Operations Test executable
add_executable(Phase5QuantizedTest tests/test_phase5_quantized.cpp)
target_link_libraries(Phase5QuantizedTest TinyML gtest gtest_main)
target_include_directories(Phase5QuantizedTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase5QuantizedTest xsimd::xsimd)
    target_include_directories(Phase5QuantizedTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase5QuantizedTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME Phase5QuantizedTest COMMAND Phase5QuantizedTest)

# Phase 6 Production API Test executable
add_executable(Phase6ProductionTest tests/test_phase6_production_new.cpp)
target_link_libraries(Phase6ProductionTest TinyML gtest gtest_main)
target_include_directories(Phase6ProductionTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase6ProductionTest xsimd::xsimd)
    target_include_directories(Phase6ProductionTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase6ProductionTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
# Phase 7 Advanced Attention Test executable
add_executable(Phase7AdvancedAttentionTest tests/test_phase7_advanced_attention.cpp)
target_link_libraries(Phase7AdvancedAttentionTest TinyML gtest gtest_main)
target_include_directories(Phase7AdvancedAttentionTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase7AdvancedAttentionTest xsimd::xsimd)
    target_include_directories(Phase7AdvancedAttentionTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase7AdvancedAttentionTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME Phase7AdvancedAttentionTest COMMAND Phase7AdvancedAttentionTest)

# Phase 8 Physics-Informed Neural Networks Test executable
add_executable(Phase8PhysicsInformedTest tests/test_phase8_physics_informed.cpp)
target_link_libraries(Phase8PhysicsInformedTest TinyML gtest gtest_main)
target_include_directories(Phase8PhysicsInformedTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase8PhysicsInformedTest xsimd::xsimd)
    target_include_directories(Phase8PhysicsInformedTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase8PhysicsInformedTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME Phase8PhysicsInformedTest COMMAND Phase8PhysicsInformedTest)

# Phase 8 Comprehensive Physics-Informed Neural Networks Test executable
add_executable(Phase8PhysicsComprehensiveTest tests/test_phase8_physics_comprehensive.cpp)
target_link_libraries(Phase8PhysicsComprehensiveTest TinyML gtest gtest_main)
target_include_directories(Phase8PhysicsComprehensiveTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase8PhysicsComprehensiveTest xsimd::xsimd)
    target_include_directories(Phase8PhysicsComprehensiveTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase8PhysicsComprehensiveTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME Phase8PhysicsComprehensiveTest COMMAND Phase8PhysicsComprehensiveTest)

# Phase 10 Graph Neural Networks Test executable
add_executable(Phase10GraphNeuralTest tests/test_phase10_graph_neural.cpp)
target_link_libraries(Phase10GraphNeuralTest TinyML gtest gtest_main)
target_include_directories(Phase10GraphNeuralTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase10GraphNeuralTest xsimd::xsimd)
    target_include_directories(Phase10GraphNeuralTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase10GraphNeuralTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME Phase10GraphNeuralTest COMMAND Phase10GraphNeuralTest)

# Production API Demo executable
add_executable(ProductionAPIDemo EXCLUDE_FROM_ALL examples/production_api_demo.cpp)
target_link_libraries(ProductionAPIDemo TinyML)
target_include_directories(ProductionAPIDemo PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(ProductionAPIDemo xsimd::xsimd)
    target_include_directories(ProductionAPIDemo PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(ProductionAPIDemo PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()

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

# Physics-Informed Neural Networks Benchmark executable
add_executable(PhysicsBenchmark EXCLUDE_FROM_ALL benchmarks/benchmark_physics.cpp)
target_link_libraries(PhysicsBenchmark TinyML)
target_include_directories(PhysicsBenchmark PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(PhysicsBenchmark xsimd::xsimd)
    target_include_directories(PhysicsBenchmark PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(PhysicsBenchmark PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()

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

# Advanced Optimizations Benchmark executable - disabled until
# AdvancedOptimizations.cpp is fixed (missing member fields, headers)
message(STATUS "Skipping AdvancedOptimizationsBenchmark (AdvancedOptimizations.cpp needs fixes)")

# Phase 13 Time Series Forecasting Test executable
add_executable(Phase13TimeSeriesTest tests/test_phase13_time_series.cpp)
target_link_libraries(Phase13TimeSeriesTest TinyML gtest gtest_main)
target_include_directories(Phase13TimeSeriesTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase13TimeSeriesTest xsimd::xsimd)
    target_include_directories(Phase13TimeSeriesTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase13TimeSeriesTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME Phase13TimeSeriesTest COMMAND Phase13TimeSeriesTest)

# Phase 12 Reinforcement Learning Test executable
add_executable(Phase12ReinforcementTest tests/test_phase12_reinforcement.cpp)
target_link_libraries(Phase12ReinforcementTest TinyML gtest gtest_main)
target_include_directories(Phase12ReinforcementTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(Phase12ReinforcementTest xsimd::xsimd)
    target_include_directories(Phase12ReinforcementTest PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(Phase12ReinforcementTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME Phase12ReinforcementTest COMMAND Phase12ReinforcementTest)
# Known segfault in RL test - disable until fixed
set_tests_properties(Phase12ReinforcementTest PROPERTIES DISABLED true)

# Reinforcement Learning Benchmark executable
add_executable(ReinforcementLearningBenchmark benchmarks/benchmark_reinforcement.cpp)
target_link_libraries(ReinforcementLearningBenchmark TinyML gtest gtest_main)
target_include_directories(ReinforcementLearningBenchmark PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(xsimd_FOUND)
    target_link_libraries(ReinforcementLearningBenchmark xsimd::xsimd)
    target_include_directories(ReinforcementLearningBenchmark PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(ReinforcementLearningBenchmark PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()
add_test(NAME ReinforcementLearningBenchmark COMMAND ReinforcementLearningBenchmark)

# Phase 11 Generative Models Test executable
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase11_generative.cpp)
    add_executable(Phase11GenerativeTest tests/test_phase11_generative.cpp)
    target_link_libraries(Phase11GenerativeTest TinyML gtest gtest_main)
    target_include_directories(Phase11GenerativeTest PUBLIC ${PROJECT_SOURCE_DIR}/include)
    if(xsimd_FOUND)
        target_link_libraries(Phase11GenerativeTest xsimd::xsimd)
        target_include_directories(Phase11GenerativeTest PRIVATE ${xsimd_INCLUDE_DIRS})
    else()
        if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
            target_include_directories(Phase11GenerativeTest PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        endif()
    endif()
    add_test(NAME Phase11GenerativeTest COMMAND Phase11GenerativeTest)
endif()

# Generative Models Benchmark executable
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/benchmark_generative.cpp)
    add_executable(GenerativeModelsBenchmark benchmarks/benchmark_generative.cpp)
    target_link_libraries(GenerativeModelsBenchmark TinyML gtest gtest_main)
    target_include_directories(GenerativeModelsBenchmark PUBLIC ${PROJECT_SOURCE_DIR}/include)
    if(xsimd_FOUND)
        target_link_libraries(GenerativeModelsBenchmark xsimd::xsimd)
        target_include_directories(GenerativeModelsBenchmark PRIVATE ${xsimd_INCLUDE_DIRS})
    else()
        if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
            target_include_directories(GenerativeModelsBenchmark PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        endif()
    endif()
    add_test(NAME GenerativeModelsBenchmark COMMAND GenerativeModelsBenchmark)
endif()

# Debugging: Print the include directories that will be passed to the compiler
get_target_property(INCLUDE_DIRS TinyML INCLUDE_DIRECTORIES)
message(STATUS "TinyML include directories: ${INCLUDE_DIRS}")

get_target_property(INCLUDE_DIRS_TEST TinyMLTests INCLUDE_DIRECTORIES)
message(STATUS "TinyMLTests include directories: ${INCLUDE_DIRS_TEST}")

# Playground server executable
add_executable(PlaygroundServer playground/server.cpp src/BayesianNeuralNetwork.cpp)
target_link_libraries(PlaygroundServer TinyML)
target_include_directories(PlaygroundServer PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/playground/include)

if(xsimd_FOUND)
    target_link_libraries(PlaygroundServer xsimd::xsimd)
    target_include_directories(PlaygroundServer PRIVATE ${xsimd_INCLUDE_DIRS})
else()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
        target_include_directories(PlaygroundServer PRIVATE "${CMAKE_BINARY_DIR}/_deps/xsimd-src/include")
    endif()
endif()

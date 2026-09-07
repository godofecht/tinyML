cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(
    TinyML
    VERSION 1.0.0
    DESCRIPTION "Lightweight C++ machine learning for embedded and real-time systems"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(TINYML_TOP_LEVEL ON)
else()
    set(TINYML_TOP_LEVEL OFF)
endif()

option(TINYML_BUILD_EXTENDED "Build the xsimd-backed extended TinyML library" ON)
option(TINYML_BUILD_TESTS "Build TinyML's test suite" ${TINYML_TOP_LEVEL})
option(TINYML_BUILD_BENCHMARKS "Build local benchmark executables" OFF)
option(TINYML_BUILD_EXAMPLES "Build example executables" OFF)
option(TINYML_BUILD_PLAYGROUND "Build the interactive playground server" OFF)
option(TINYML_FETCH_DEPENDENCIES "Fetch missing third-party build dependencies" ${TINYML_TOP_LEVEL})
option(TINYML_ENABLE_WARNINGS "Enable compiler warnings for TinyML targets" ${TINYML_TOP_LEVEL})
option(TINYML_ENABLE_LTO "Enable interprocedural optimization when supported" OFF)

if(TINYML_BUILD_BENCHMARKS AND NOT TINYML_BUILD_EXTENDED)
    message(FATAL_ERROR "TINYML_BUILD_BENCHMARKS requires TINYML_BUILD_EXTENDED=ON")
endif()

if(TINYML_BUILD_EXAMPLES AND NOT TINYML_BUILD_EXTENDED)
    message(FATAL_ERROR "TINYML_BUILD_EXAMPLES requires TINYML_BUILD_EXTENDED=ON")
endif()

if(TINYML_BUILD_PLAYGROUND AND NOT TINYML_BUILD_EXTENDED)
    message(FATAL_ERROR "TINYML_BUILD_PLAYGROUND requires TINYML_BUILD_EXTENDED=ON")
endif()

function(tinyml_configure_target target)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_include_directories(
        ${target}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    set_target_properties(
        ${target}
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            WINDOWS_EXPORT_ALL_SYMBOLS ON
    )

    if(TINYML_ENABLE_WARNINGS)
        if(MSVC)
            target_compile_options(${target} PRIVATE /W4 /permissive-)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()

    if(TINYML_ENABLE_LTO)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
        if(ipo_supported)
            set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        else()
            message(WARNING "IPO/LTO requested for ${target}, but unsupported: ${ipo_error}")
        endif()
    endif()
endfunction()

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/generated/tinyml")
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/TinyMLVersion.hpp.in"
    "${PROJECT_BINARY_DIR}/generated/tinyml/version.hpp"
    @ONLY
)

set(TINYML_CORE_SOURCES
    src/NN.cpp
    src/Network.cpp
)

set(TINYML_CORE_HEADERS
    include/NN.h
    include/Network.h
    include/Model.h
)

add_library(TinyMLCore ${TINYML_CORE_SOURCES})
add_library(TinyML::Core ALIAS TinyMLCore)
set_target_properties(
    TinyMLCore
    PROPERTIES
        EXPORT_NAME Core
        OUTPUT_NAME tinyml-core
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)
tinyml_configure_target(TinyMLCore)

set(TINYML_EXTENDED_HEADERS
    include/AdvancedAttention.h
    include/DynamicNeuralNetwork.h
    include/GenerativeModels.h
    include/GraphNeuralNetwork.h
    include/LightweightAttention.h
    include/PhysicsInformedNN.h
    include/ProductionAPI.h
    include/QuantizedOperations.h
    include/RealTimeTransformer.h
    include/ReinforcementLearning.h
    include/TimeSeriesForecasting.h
    include/TinyMLAPI.h
    include/XSIMDOperations.h
)

if(TINYML_BUILD_EXTENDED)
    find_package(xsimd 12.1 QUIET CONFIG)

    if(TARGET xsimd)
        set(TINYML_XSIMD_TARGET xsimd)
    elseif(TARGET xsimd::xsimd)
        set(TINYML_XSIMD_TARGET xsimd::xsimd)
    else()
        if(NOT TINYML_FETCH_DEPENDENCIES)
            message(FATAL_ERROR
                "TinyML extended requires xsimd >= 12.1. Install xsimd or configure with "
                "-DTINYML_FETCH_DEPENDENCIES=ON. Core-only builds never require xsimd."
            )
        endif()

        include(FetchContent)
        set(XSIMD_SKIP_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            xsimd
            URL https://github.com/xtensor-stack/xsimd/archive/refs/tags/12.1.1.tar.gz
        )
        FetchContent_MakeAvailable(xsimd)

        if(TARGET xsimd)
            set(TINYML_XSIMD_TARGET xsimd)
        elseif(TARGET xsimd::xsimd)
            set(TINYML_XSIMD_TARGET xsimd::xsimd)
        else()
            message(FATAL_ERROR "xsimd was fetched but did not provide a supported CMake target")
        endif()
    endif()

    set(TINYML_EXTENDED_SOURCES
        src/AdvancedAttention.cpp
        src/DynamicNeuralNetwork.cpp
        src/GenerativeModels.cpp
        src/GraphNeuralNetwork.cpp
        src/LightweightAttention.cpp
        src/PDESolvers.cpp
        src/PhysicsInformedNN.cpp
        src/ProductionAPI.cpp
        src/QuantizedOperations.cpp
        src/RealTimeTransformer.cpp
        src/ReinforcementLearning.cpp
        src/TimeSeriesForecasting.cpp
        src/TinyMLAPI.cpp
        src/TinyMLUtils.cpp
        src/XSIMDOperations.cpp
    )

    add_library(TinyML ${TINYML_EXTENDED_SOURCES})
    add_library(TinyML::Extended ALIAS TinyML)
    add_library(TinyML::TinyML ALIAS TinyML)
    set_target_properties(
        TinyML
        PROPERTIES
            EXPORT_NAME Extended
            OUTPUT_NAME tinyml
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    tinyml_configure_target(TinyML)
    target_link_libraries(
        TinyML
        PUBLIC
            $<BUILD_INTERFACE:TinyMLCore>
            $<INSTALL_INTERFACE:TinyML::Core>
            $<BUILD_INTERFACE:${TINYML_XSIMD_TARGET}>
            $<INSTALL_INTERFACE:xsimd>
    )

    if(UNIX AND NOT APPLE)
        target_link_libraries(TinyML PRIVATE m)
    endif()
endif()

if((TINYML_BUILD_TESTS AND TINYML_BUILD_EXTENDED) OR TINYML_BUILD_BENCHMARKS)
    find_package(GTest QUIET)

    if(NOT TARGET GTest::gtest_main)
        if(NOT TINYML_FETCH_DEPENDENCIES)
            message(FATAL_ERROR
                "TinyML tests/benchmarks require GoogleTest. Install it or configure with "
                "-DTINYML_FETCH_DEPENDENCIES=ON."
            )
        endif()

        include(FetchContent)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        )
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

if(TINYML_BUILD_TESTS)
    enable_testing()

    add_executable(TinyMLCoreSmoke tests/core_smoke.cpp)
    target_link_libraries(TinyMLCoreSmoke PRIVATE TinyMLCore)
    target_compile_features(TinyMLCoreSmoke PRIVATE cxx_std_17)
    add_test(NAME TinyMLCoreSmoke COMMAND TinyMLCoreSmoke)

    if(TINYML_BUILD_EXTENDED)
        function(tinyml_add_gtest target)
            add_executable(${target} ${ARGN})
            target_link_libraries(${target} PRIVATE TinyML GTest::gtest_main)
            target_compile_features(${target} PRIVATE cxx_std_17)
            add_test(NAME ${target} COMMAND ${target})
        endfunction()

        tinyml_add_gtest(
            TinyMLTests
            tests/test_phase1_simd.cpp
            tests/test_phase3_dynamic_new.cpp
            tests/test_phase4_fixed.cpp
            tests/test_phase4_simple.cpp
            tests/test_phase5_quantized.cpp
            tests/test_phase6_production.cpp
            tests/test_phase7_advanced_attention.cpp
            tests/test_phase11_generative.cpp
        )
        tinyml_add_gtest(BasicTests tests/basic_tests.cpp)
        tinyml_add_gtest(Phase4FixedTest tests/test_phase4_fixed.cpp)
        tinyml_add_gtest(Phase5QuantizedTest tests/test_phase5_quantized.cpp)
        tinyml_add_gtest(Phase7AdvancedAttentionTest tests/test_phase7_advanced_attention.cpp)
        tinyml_add_gtest(Phase8PhysicsInformedTest tests/test_phase8_physics_informed.cpp)
        tinyml_add_gtest(Phase8PhysicsComprehensiveTest tests/test_phase8_physics_comprehensive.cpp)
        tinyml_add_gtest(Phase10GraphNeuralTest tests/test_phase10_graph_neural.cpp)
        tinyml_add_gtest(LightweightAttentionTest tests/test_lightweight_attention.cpp)
        tinyml_add_gtest(Phase13TimeSeriesTest tests/test_phase13_time_series.cpp)
        tinyml_add_gtest(Phase12ReinforcementTest tests/test_phase12_reinforcement.cpp)
        tinyml_add_gtest(Phase11GenerativeTest tests/test_phase11_generative.cpp)

        set_tests_properties(
            Phase8PhysicsInformedTest
            PROPERTIES
                DISABLED TRUE
                LABELS "long-running"
        )
        set_tests_properties(
            Phase8PhysicsComprehensiveTest
            PROPERTIES
                DISABLED TRUE
                LABELS "flaky"
        )
        set_tests_properties(
            Phase12ReinforcementTest
            PROPERTIES
                DISABLED TRUE
                LABELS "known-failure"
        )
    endif()
endif()

if(TINYML_BUILD_BENCHMARKS)
    add_executable(SIMDBenchmark benchmarks/benchmark_xsimd.cpp)
    target_link_libraries(SIMDBenchmark PRIVATE TinyML GTest::gtest_main)

    add_executable(AttentionBenchmark benchmarks/benchmark_attention.cpp)
    target_link_libraries(AttentionBenchmark PRIVATE TinyML GTest::gtest_main)

    add_executable(SimpleAttentionBenchmark benchmarks/benchmark_simple_attention.cpp)
    target_include_directories(SimpleAttentionBenchmark PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_features(SimpleAttentionBenchmark PRIVATE cxx_std_17)

    add_executable(PhysicsBenchmark benchmarks/benchmark_physics.cpp)
    target_link_libraries(PhysicsBenchmark PRIVATE TinyML)

    add_executable(GenerativeModelsBenchmark benchmarks/benchmark_generative.cpp)
    target_link_libraries(GenerativeModelsBenchmark PRIVATE TinyML GTest::gtest_main)
endif()

if(TINYML_BUILD_EXAMPLES)
    add_executable(ProductionAPIDemo examples/production_api_demo.cpp)
    target_link_libraries(ProductionAPIDemo PRIVATE TinyML)
endif()

if(TINYML_BUILD_PLAYGROUND)
    add_executable(
        PlaygroundServer
        playground/server.cpp
        src/BayesianNeuralNetwork.cpp
    )
    target_link_libraries(PlaygroundServer PRIVATE TinyML)
    target_include_directories(PlaygroundServer PRIVATE ${PROJECT_SOURCE_DIR}/playground/include)
endif()

install(
    TARGETS TinyMLCore
    EXPORT TinyMLCoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    FILES ${TINYML_CORE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    FILES include/tinyml/core.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tinyml
)
install(
    FILES "${PROJECT_BINARY_DIR}/generated/tinyml/version.hpp"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tinyml
)

install(
    EXPORT TinyMLCoreTargets
    FILE TinyMLCoreTargets.cmake
    NAMESPACE TinyML::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TinyML
)

if(TINYML_BUILD_EXTENDED)
    install(
        TARGETS TinyML
        EXPORT TinyMLExtendedTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(
        FILES ${TINYML_EXTENDED_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(
        FILES include/tinyml/tinyml.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tinyml
    )
    install(
        EXPORT TinyMLExtendedTargets
        FILE TinyMLExtendedTargets.cmake
        NAMESPACE TinyML::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TinyML
    )
endif()

configure_package_config_file(
    cmake/TinyMLConfig.cmake.in
    "${PROJECT_BINARY_DIR}/TinyMLConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TinyML
)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/TinyMLConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${PROJECT_BINARY_DIR}/TinyMLConfig.cmake"
        "${PROJECT_BINARY_DIR}/TinyMLConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TinyML
)

install(
    FILES README.md LICENSE LICENSING.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/TinyML
)
if(TINYML_BUILD_EXTENDED)
    install(
        FILES LICENSE-COMMERCIAL.md
        DESTINATION ${CMAKE_INSTALL_DATADIR}/TinyML
    )
endif()

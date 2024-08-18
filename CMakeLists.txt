# CMake version and project name
cmake_minimum_required(VERSION 3.10)
project(TinyML VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set output directories for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add the include directory so that the compiler can find header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Collect all source files from the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create the main library target
add_library(TinyML ${SOURCES})

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

# Create test executable
add_executable(TinyMLTests ${TEST_SOURCES})

# Link the test executable with Google Test and your library
target_link_libraries(TinyMLTests TinyML gtest gtest_main)

# Add test to CMake's testing framework
add_test(NAME TinyMLTests COMMAND TinyMLTests)

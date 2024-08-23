# CMake version and project name
cmake_minimum_required(VERSION 3.10)
project(TinyML VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set output directories for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Collect all source files from the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Print out the source files collected
message(STATUS "Source files: ${SOURCES}")

# Create the main library target
add_library(TinyML ${SOURCES})

# Ensure that the include directories for the library are available to targets that link with the library
target_include_directories(TinyML PUBLIC ${PROJECT_SOURCE_DIR}/include)

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

# Print the include directories being set for the test target
message(STATUS "Include directory for TinyMLTests: ${PROJECT_SOURCE_DIR}/include")

# Add test to CMake's testing framework
add_test(NAME TinyMLTests COMMAND TinyMLTests)

# Debugging: Print the include directories that will be passed to the compiler
get_target_property(INCLUDE_DIRS TinyML INCLUDE_DIRECTORIES)
message(STATUS "TinyML include directories: ${INCLUDE_DIRS}")

get_target_property(INCLUDE_DIRS_TEST TinyMLTests INCLUDE_DIRECTORIES)
message(STATUS "TinyMLTests include directories: ${INCLUDE_DIRS_TEST}")

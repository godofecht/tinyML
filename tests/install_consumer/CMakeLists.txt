cmake_minimum_required(VERSION 3.20)

project(TinyMLInstallConsumer LANGUAGES CXX)

find_package(TinyML 1.0 CONFIG REQUIRED COMPONENTS Core)

add_executable(tinyml_install_consumer main.cpp)
target_link_libraries(tinyml_install_consumer PRIVATE TinyML::Core)
target_compile_features(tinyml_install_consumer PRIVATE cxx_std_17)

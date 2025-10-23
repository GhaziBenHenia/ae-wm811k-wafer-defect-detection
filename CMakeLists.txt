cmake_minimum_required(VERSION 3.16)
project(ftg_preprocess LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(preprocess preprocess.cpp)

find_package(opencv REQUIRED CONFIG)
target_link_libraries(preprocess PRIVATE opencv::opencv)

target_link_options(preprocess PRIVATE "/FORCE:MULTIPLE")

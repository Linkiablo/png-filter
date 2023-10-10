cmake_minimum_required(VERSION 3.27)

project(png_filter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -march=native -Wall -Werror -Wextra -Wpedantic -fsanitize=address -fsanitize=leak")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(png_filter main.cpp)

target_link_libraries(png_filter libpng.so)

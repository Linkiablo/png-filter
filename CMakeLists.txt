cmake_minimum_required(VERSION 3.27)

project(png_filter)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

string(TOLOWER ${CMAKE_BUILD_TYPE} profile)
if(profile STREQUAL "debug")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_OMP -fopenmp -ggdb -march=native -Wall -Wextra -Wpedantic -fsanitize=address -fsanitize=leak")
elseif(profile STREQUAL "release")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -D_USE_OMP -fopenmp -march=native")
elseif(profile STREQUAL "bench")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -D_BENCH -D_USE_OMP -fopenmp -march=native")

	find_package(benchmark REQUIRED)

	add_executable(bench bench.cpp main.cpp)
	target_link_libraries(bench libpng.so)
	target_link_libraries(bench libgomp.so)
	target_link_libraries(bench benchmark::benchmark)

	return()
endif()

add_executable(png_filter main.cpp)

target_compile_definitions(png_filter PRIVATE
  $<$<CONFIG:Debug>:DEBUG_BUILD>
)

target_link_libraries(png_filter libpng.so)
target_link_libraries(png_filter libgomp.so)

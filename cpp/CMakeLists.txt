cmake_minimum_required(VERSION 3.27)

project(png_filter)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_library(LIBASAN NAMES "asan" "libasan" "libasan.so" "libasan.so.8")
if(NOT LIBASAN)
	message(FATAL_ERROR "libasan not found!")
endif()

find_library(LIBPNG NAMES "png" "libpng" "libpng.so")
if(NOT LIBPNG)
	message(FATAL_ERROR "libpng not found!")
endif()

find_library(LIBGOMP NAMES "gomp" "libgomp" "libgomp.so" "libgomp.so.1")
if(NOT LIBGOMP)
	message(FATAL_ERROR "libgomp not found!")
endif()

find_package(benchmark REQUIRED)

add_executable(png_filter main.cpp)

set(DEBUG_FLAGS "-fopenmp" "-ggdb" "-march=native" "-Wall" "-Wextra" "-Wpedantic"
	"-fsanitize=address" "-fsanitize=leak")
set(RELEASE_FLAGS "-fopenmp" "-march=native" "-O2")
target_compile_options(png_filter PUBLIC
	$<$<CONFIG:Debug>:${DEBUG_FLAGS}>
	$<$<CONFIG:Release>:${RELEASE_FLAGS}>
	$<$<CONFIG:Bench>:${RELEASE_FLAGS}>
)

target_compile_definitions(png_filter PUBLIC
	$<$<CONFIG:Bench>:"_BENCH">
)

target_link_libraries(png_filter
	$<$<CONFIG:Debug>:${LIBASAN} ${LIBPNG} ${LIBGOMP}>
	$<$<CONFIG:Release>:${LIBPNG} ${LIBGOMP}>
	$<$<CONFIG:Bench>:benchmark::benchmark ${LIBPNG} ${LIBGOMP}>
)

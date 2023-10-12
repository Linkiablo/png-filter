#include <benchmark/benchmark.h>

#include "image.hpp"

static void BM_standard_filter(benchmark::State &state) {
    auto image = read_png("wave.png");

    for (auto _ : state)
	    image.apply_nearest_filter(1, 1);
}
BENCHMARK(BM_standard_filter);

static void BM_simd_filter(benchmark::State &state) {
    auto image = read_png("wave.png");

    for (auto _ : state)
	    image.apply_nearest_filter_simd(1, 1);
}
BENCHMARK(BM_simd_filter);

static void BM_omp_filter(benchmark::State &state) {
    auto image = read_png("wave.png");

    for (auto _ : state)
	    image.apply_nearest_filter_omp(1, 1);
}
BENCHMARK(BM_omp_filter);

BENCHMARK_MAIN();

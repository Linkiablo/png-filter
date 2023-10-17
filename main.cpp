#ifdef _BENCH
#include <benchmark/benchmark.h>
#endif
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <png.h>
#include <ranges>
#include <vector>

#include "image.hpp"

void Image::apply_nearest_filter(int32_t nx, int32_t ny) {
    auto new_width = this->_width - 2 * nx;
    auto new_height = this->_height - 2 * ny;

    std::vector<Pixel3> filtered_buf(new_width * new_height,
                                     {.r = 0, .g = 0, .b = 0});

    auto count = (nx * 2 + 1) * (ny * 2 + 1);

    auto lines = this->_buf | std::views::chunk(this->_width);

    for (int64_t y = ny; y < this->_height - ny; ++y) {
        for (int64_t x = nx; x < this->_width - nx; ++x) {
            uint32_t sum_r = 0;
            uint32_t sum_g = 0;
            uint32_t sum_b = 0;

            for (int64_t yy = y - ny; yy <= y + ny; ++yy) {
                auto cur_line = lines[yy];

                for (int64_t xx = x - nx; xx <= x + nx; ++xx) {
                    sum_r += cur_line[xx].r;
                    sum_g += cur_line[xx].g;
                    sum_b += cur_line[xx].b;
                }
            }

            filtered_buf.at((y - ny) * new_width + (x - nx)).r = sum_r / count;
            filtered_buf.at((y - ny) * new_width + (x - nx)).g = sum_g / count;
            filtered_buf.at((y - ny) * new_width + (x - nx)).b = sum_b / count;
        }
    }

    this->_width = new_width;
    this->_height = new_height;

    this->_buf = filtered_buf;
}

void Image::apply_nearest_filter_simd(int32_t nx, int32_t ny) {
    auto new_width = this->_width - 2 * nx;
    auto new_height = this->_height - 2 * ny;

    std::vector<Pixel3> filtered_buf(new_width * new_height,
                                     {.r = 0, .g = 0, .b = 0});

    auto lines = this->_buf | std::views::chunk(this->_width);

    for (int64_t y = ny; y < this->_height - ny; ++y) {
        for (int64_t x = nx; x < this->_width - nx; ++x) {
            Pixel3Sum sum = Pixel3Sum();

            for (int64_t yy = y - ny; yy <= y + ny; ++yy) {

                auto cur_line = lines[yy];

                for (int64_t xx = x - nx; xx <= x + nx; ++xx) {

                    sum += cur_line[xx];
                }
            }

            filtered_buf.at((y - ny) * new_width + (x - nx)) = sum.to_pixel_3();
        }
    }

    this->_width = new_width;
    this->_height = new_height;

    this->_buf = filtered_buf;
}

void Image::apply_nearest_filter_omp(int32_t nx, int32_t ny) {
    auto new_width = this->_width - 2 * nx;
    auto new_height = this->_height - 2 * ny;

    std::vector<Pixel3> filtered_buf(new_width * new_height,
                                     {.r = 0, .g = 0, .b = 0});

    auto count = (nx * 2 + 1) * (ny * 2 + 1);

    auto lines = this->_buf | std::views::chunk(this->_width);

#pragma omp parallel for
    for (int64_t y = ny; y < this->_height - ny; ++y) {
        for (int64_t x = nx; x < this->_width - nx; ++x) {
            uint32_t sum_r = 0;
            uint32_t sum_g = 0;
            uint32_t sum_b = 0;

            for (int64_t yy = y - ny; yy <= y + ny; ++yy) {
                auto cur_line = lines[yy];

                for (int64_t xx = x - nx; xx <= x + nx; ++xx) {
                    sum_r += cur_line[xx].r;
                    sum_g += cur_line[xx].g;
                    sum_b += cur_line[xx].b;
                }
            }

            filtered_buf.at((y - ny) * new_width + (x - nx)).r = sum_r / count;
            filtered_buf.at((y - ny) * new_width + (x - nx)).g = sum_g / count;
            filtered_buf.at((y - ny) * new_width + (x - nx)).b = sum_b / count;
        }
    }
    this->_width = new_width;
    this->_height = new_height;

    this->_buf = filtered_buf;
}

#ifdef _BENCH
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

#else
int main() {
    auto image = read_png("wave.png");

    image.apply_nearest_filter(2, 2);

    write_png("wave_cp.png", image);

    return 0;
}
#endif

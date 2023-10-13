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
    std::vector<Pixel3> filtered_buf = this->_buf;

    auto lines = this->_buf | std::views::chunk(this->_width);

    for (int64_t y = 0; y < this->_height; ++y) {
        for (int64_t x = 0; x < this->_width; ++x) {
            uint32_t sum_r = 0;
            uint32_t sum_g = 0;
            uint32_t sum_b = 0;

            uint32_t count = 0;

            for (int64_t yy = std::max((int64_t)0, y - ny);
                 yy < std::min((int64_t)this->_height, y + ny); ++yy) {

                auto cur_line = lines[yy];

                for (int64_t xx = std::max((int64_t)0, x - nx);
                     xx < std::min((int64_t)this->_width, x + nx); ++xx) {

                    sum_r += cur_line[xx].r;
                    sum_g += cur_line[xx].g;
                    sum_b += cur_line[xx].b;

                    ++count;
                }
            }

            filtered_buf[y * this->_width + x].r = sum_r / count;
            filtered_buf[y * this->_width + x].g = sum_g / count;
            filtered_buf[y * this->_width + x].b = sum_b / count;
        }
    }

    this->_buf = filtered_buf;
}

void Image::apply_nearest_filter_simd(int32_t nx, int32_t ny) {
    std::vector<Pixel3> filtered_buf = this->_buf;

    auto lines = this->_buf | std::views::chunk(this->_width);

    for (int64_t y = 0; y < this->_height; ++y) {
        for (int64_t x = 0; x < this->_width; ++x) {
            Pixel3Sum sum = Pixel3Sum();

            for (int64_t yy = std::max((int64_t)0, y - ny);
                 yy < std::min((int64_t)this->_height, y + ny); ++yy) {

                auto cur_line = lines[yy];

                for (int64_t xx = std::max((int64_t)0, x - nx);
                     xx < std::min((int64_t)this->_width, x + nx); ++xx) {

                    sum += cur_line[xx];
                }
            }

            filtered_buf[y * this->_width + x] = sum.to_pixel_3();
        }
    }

    this->_buf = filtered_buf;
}

void Image::apply_nearest_filter_omp(int32_t nx, int32_t ny) {
    std::vector<Pixel3> filtered_buf = this->_buf;

    auto lines = this->_buf | std::views::chunk(this->_width);

#pragma omp parallel for
    for (int64_t y = 0; y < this->_height; ++y) {
        for (int64_t x = 0; x < this->_width; ++x) {
            uint32_t sum_r = 0;
            uint32_t sum_g = 0;
            uint32_t sum_b = 0;

            uint32_t count = 0;

            for (int64_t yy = std::max((int64_t)0, y - ny);
                 yy < std::min((int64_t)this->_height, y + ny); ++yy) {

                auto cur_line = lines[yy];

                for (int64_t xx = std::max((int64_t)0, x - nx);
                     xx < std::min((int64_t)this->_width, x + nx); ++xx) {

                    sum_r += cur_line[xx].r;
                    sum_g += cur_line[xx].g;
                    sum_b += cur_line[xx].b;

                    ++count;
                }
            }

            filtered_buf[y * this->_width + x].r = sum_r / count;
            filtered_buf[y * this->_width + x].g = sum_g / count;
            filtered_buf[y * this->_width + x].b = sum_b / count;
        }
    }

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

    image.apply_nearest_filter_simd(3, 3);

    write_png("wave_cp.png", image);

    // auto ps = Pixel3Sum();

    // ps += Pixel3{2, 3, 4};
    // ps += Pixel3{4, 6, 8};

    // auto p = ps.to_pixel_3();

    // assert(p.r == 3);
    // assert(p.g == 4);
    // assert(p.b == 6);

    return 0;
}
#endif

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

    for (int64_t y = ny; y < this->_height - ny; ++y) {
        for (int64_t x = nx; x < this->_width - nx; ++x) {
            uint32_t sum_r = 0;
            uint32_t sum_g = 0;
            uint32_t sum_b = 0;

            for (int64_t yy = y - ny; yy <= y + ny; ++yy) {
                auto cur_line_offset = yy * this->_width;

                for (int64_t xx = x - nx; xx <= x + nx; ++xx) {
                    sum_r += this->_buf[cur_line_offset + xx].r;
                    sum_g += this->_buf[cur_line_offset + xx].g;
                    sum_b += this->_buf[cur_line_offset + xx].b;
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
    for (int64_t y = ny; y < this->_height - ny; ++y) {
        for (int64_t x = nx; x < this->_width - nx; ++x) {
            Pixel3Sum sum = Pixel3Sum();

            for (int64_t yy = y - ny; yy <= y + ny; ++yy) {

                auto cur_line_offset = yy * this->_width;

                for (int64_t xx = x - nx; xx <= x + nx; ++xx) {

                    sum += this->_buf[cur_line_offset + xx];
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

#pragma omp parallel for
    for (int64_t y = ny; y < this->_height - ny; ++y) {
        for (int64_t x = nx; x < this->_width - nx; ++x) {
            uint32_t sum_r = 0;
            uint32_t sum_g = 0;
            uint32_t sum_b = 0;

            for (int64_t yy = y - ny; yy <= y + ny; ++yy) {
                auto cur_line_offset = yy * this->_width;

                for (int64_t xx = x - nx; xx <= x + nx; ++xx) {
                    sum_r += this->_buf[cur_line_offset + xx].r;
                    sum_g += this->_buf[cur_line_offset + xx].g;
                    sum_b += this->_buf[cur_line_offset + xx].b;
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

#ifndef _BENCH
int main() {
    auto image = read_png("wave.png");

    image.apply_nearest_filter(2, 2);

    write_png("wave_cp.png", image);

    return 0;
}
#endif

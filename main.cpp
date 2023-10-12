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

Image read_png(const char *filename) {
    FILE *infile = fopen(filename, "rb");

    png_structp png_ptr =
        png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);

    png_init_io(png_ptr, infile);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    Pixel3 **raw_pixel_data =
        reinterpret_cast<Pixel3 **>(png_get_rows(png_ptr, info_ptr));

    uint32_t width = png_get_image_width(png_ptr, info_ptr);
    uint32_t height = png_get_image_height(png_ptr, info_ptr);

    std::vector<Pixel3> pixel_data;
    pixel_data.reserve(width * height);
    for (uint32_t i = 0; i < height; ++i) {
        pixel_data.insert(pixel_data.end(), raw_pixel_data[i],
                          raw_pixel_data[i] + width);
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(infile);

    return Image{width, height, pixel_data};
}

void write_png(const char *filename, const Image &image) {
    FILE *fp = fopen(filename, "wb");
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    uint32_t width = image._width;
    uint32_t height = image._height;

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    // entfernen fuer default
    png_set_compression_level(png_ptr, 0);

    png_init_io(png_ptr, fp);

    uint8_t **row_pointers = new uint8_t *[height];
    for (uint32_t i = 0; i < image._buf.size(); i += width) {
        uint32_t ii = i / width;
        Pixel3 const *p = &image._buf[0] + i;
        row_pointers[ii] = reinterpret_cast<uint8_t *>(const_cast<Pixel3 *>(p));
    }
    png_set_rows(png_ptr, info_ptr, row_pointers);

    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);
    delete[] row_pointers;
    fclose(fp);
}

#ifndef _BENCH
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

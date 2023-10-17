#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <png.h>
#include <stdint.h>
#include <vector>

typedef struct Pixel3 {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} Pixel3;

typedef struct Pixel3Sum {
    __m128 inner;

    Pixel3Sum() { this->inner = _mm_setzero_ps(); }

    Pixel3Sum &operator+=(const Pixel3 &rhs) {
        // auto other = _mm_setr_epi32(rhs.r, rhs.g, rhs.b, 1);
        auto other = _mm_set_ps(rhs.r, rhs.g, rhs.b, 1);
        this->inner = _mm_add_ps(this->inner, other);

        return *this;
    }

    Pixel3 to_pixel_3() {
        // uint32_t count = ((uint32_t *)&(this->inner))[3];

        // auto tmp = _mm_cvtepi32_ps(this->inner);
        auto packed_count = _mm_set1_ps(((float *)&(this->inner))[0]);
        this->inner = _mm_div_ps(this->inner, packed_count);
        auto inner_int = _mm_cvtps_epi32(this->inner);

        uint8_t r = ((uint32_t *)&(inner_int))[3];
        uint8_t g = ((uint32_t *)&(inner_int))[2];
        uint8_t b = ((uint32_t *)&(inner_int))[1];

        auto res = Pixel3{r, g, b};

        return res;
    }
} Pixel3Sum;

class Image {
  public:
    Image(uint32_t width, uint32_t height, std::vector<Pixel3> &buf)
        : _width(width), _height(height), _buf(buf){};

    void apply_nearest_filter(int32_t nx, int32_t ny);
    void apply_nearest_filter_avx(int32_t nx, int32_t ny);
    void apply_nearest_filter_omp(int32_t nx, int32_t ny);
    void apply_nearest_filter_simd(int32_t nx, int32_t ny);

    uint32_t _width;
    uint32_t _height;
    std::vector<Pixel3> _buf;
};

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

#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <stdint.h>
#include <vector>

typedef struct Pixel3 {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} Pixel3;

typedef struct Pixel3Sum {

    __m128i inner;

    Pixel3Sum() { this->inner = _mm_setzero_si128(); }

    Pixel3Sum &operator+=(const Pixel3 &rhs) {
        auto other = _mm_setr_epi32(rhs.r, rhs.g, rhs.b, 1);
        this->inner = _mm_add_epi32(this->inner, other);

        return *this;
    }

    Pixel3 to_pixel_3() {
        uint32_t count = ((uint32_t *)&(this->inner))[3];

        uint8_t r = static_cast<uint8_t>(((uint32_t *)&(this->inner))[0] / count);
        uint8_t g = static_cast<uint8_t>(((uint32_t *)&(this->inner))[1] / count);
        uint8_t b = static_cast<uint8_t>(((uint32_t *)&(this->inner))[2] / count);

        auto res = Pixel3{r, g, b};

        return res;
    }
} Pixel3Sum;

class Image {
  public:
    Image(uint32_t width, uint32_t height, std::vector<Pixel3> &buf)
        : _width(width), _height(height), _buf(buf){};

    void apply_nearest_filter(int32_t nx, int32_t ny);
    void apply_nearest_filter_omp(int32_t nx, int32_t ny);
    void apply_nearest_filter_simd(int32_t nx, int32_t ny);

    uint32_t _width;
    uint32_t _height;
    std::vector<Pixel3> _buf;
};

Image read_png(const char *filename);

void write_png(const char *filename, const Image &image);

#![feature(portable_simd)]
#![feature(array_chunks)]
#![feature(test)]
extern crate test;

use std::simd::Simd;
use std::{fs, io, path::Path};

struct Image {
    color: png::ColorType,
    width: u32,
    height: u32,
    inner: Vec<u8>,
}

impl Image {
    fn from_png(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let decoder = png::Decoder::new(fs::File::open(path)?);
        let mut decoder = decoder.read_info()?;

        let mut inner = vec![0; decoder.output_buffer_size()];

        let info = decoder.next_frame(&mut inner)?;

        let width = info.width;
        let height = info.height;

        let color = info.color_type;

        Ok(Self {
            color,
            width,
            height,
            inner,
        })
    }

    fn write_png(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let file = fs::File::create(path)?;
        let bw = io::BufWriter::new(file);

        let mut encoder = png::Encoder::new(bw, self.width, self.height);

        encoder.set_color(self.color);

        let mut writer = encoder.write_header()?;

        writer.write_image_data(&self.inner)?;

        Ok(())
    }

    fn apply_nearest_filter(&mut self, nx: u32, ny: u32) {
        let new_width = self.width - 2 * nx;
        let new_height = self.height - 2 * ny;

        let count = (nx * 2 + 1) * (ny * 2 + 1);

        let mut filtered_buf: Vec<u8> = vec![0; (new_width * new_height * 3) as usize];

        for y in ny..self.height - ny {
            for x in nx..self.width - nx {
                let mut sum_r = 0;
                let mut sum_g = 0;
                let mut sum_b = 0;

                for yy in y - ny..=y + ny {
                    let cur_offset = yy * self.width * 3;

                    self.inner[(cur_offset + (x - nx) * 3) as usize
                        ..=(cur_offset + (x + nx) * 3 + 2) as usize]
                        .array_chunks::<3>()
                        .for_each(|c| {
                            sum_r += c[0] as u32;
                            sum_g += c[1] as u32;
                            sum_b += c[2] as u32;
                        });
                }

                filtered_buf[(((y - ny) * new_width + (x - nx)) * 3) as usize] =
                    (sum_r / count) as u8;
                filtered_buf[(((y - ny) * new_width + (x - nx)) * 3 + 1) as usize] =
                    (sum_g / count) as u8;
                filtered_buf[(((y - ny) * new_width + (x - nx)) * 3 + 2) as usize] =
                    (sum_b / count) as u8;
            }
        }

        self.width = new_width;
        self.height = new_height;
        self.inner = filtered_buf;
    }

    fn apply_nearest_filter_simd(&mut self, nx: u32, ny: u32) {
        let new_width = self.width - 2 * nx;
        let new_height = self.height - 2 * ny;

        let count = (nx * 2 + 1) * (ny * 2 + 1);
        let simd_count = Simd::from_array([count; 4]);

        let mut filtered_buf: Vec<u8> = vec![0; (new_width * new_height * 3) as usize];

        for y in ny..self.height - ny {
            for x in nx..self.width - nx {
                let mut sum = Simd::<u32, 4>::from_array([0; 4]);

                for yy in y - ny..=y + ny {
                    let cur_offset = yy * self.width * 3;

                    self.inner[(cur_offset + (x - nx) * 3) as usize
                        ..=(cur_offset + (x + nx) * 3 + 2) as usize]
                        .array_chunks::<3>()
                        .for_each(|c| {
                            let c = c.map(|b| b as u32);

                            let sc = Simd::<u32, 4>::gather_or_default(
                                &c,
                                Simd::from_array([0, 1, 2, 10]),
                            );

                            sum += sc;
                        });
                }

                sum /= simd_count;

                filtered_buf[(((y - ny) * new_width + (x - nx)) * 3) as usize] = sum[0] as u8;
                filtered_buf[(((y - ny) * new_width + (x - nx)) * 3 + 1) as usize] = sum[1] as u8;
                filtered_buf[(((y - ny) * new_width + (x - nx)) * 3 + 2) as usize] = sum[2] as u8;
            }
        }

        self.width = new_width;
        self.height = new_height;
        self.inner = filtered_buf;
    }
}

fn main() {
    let mut image = Image::from_png("wave.png").unwrap();

    image.apply_nearest_filter(5, 5);

    image.write_png("wave_cp.png").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn same_output() {
        let mut image = Image::from_png("wave.png").unwrap();
        let mut image_simd = Image::from_png("wave.png").unwrap();

        image.apply_nearest_filter(1, 1);
        image_simd.apply_nearest_filter(1, 1);

        assert_eq!(image.inner, image_simd.inner);
    }

    #[bench]
    fn bench_standard_filter(b: &mut Bencher) {
        let mut image = Image::from_png("wave.png").unwrap();

        b.iter(|| {
            image.apply_nearest_filter(1, 1);
        });
    }

    #[bench]
    fn bench_simd_filter(b: &mut Bencher) {
        let mut image = Image::from_png("wave.png").unwrap();

        b.iter(|| {
            image.apply_nearest_filter_simd(1, 1);
        });
    }
}

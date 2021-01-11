use std::env;

use apodize;
use color_space;
use ebur128::{self, EbuR128};
use hound;
use image::{ImageBuffer, RgbImage};
use opencv::{
    self,
    core::*,
    highgui, imgcodecs, imgproc,
    prelude::*,
    types::*,
    videoio::{VideoWriter, VideoWriterTrait},
};
use rand::{self, rngs::ThreadRng, Rng};
use rustfft::{self, num_complex::Complex32};
use vulkano;
use vulkano_shaders;

mod util;

const GAIN_0: f32 = 0.708;
const FFT_SIZE: usize = 65536 / 4;
//const IMAGE_HEIGHT: u32 = 200;
const PITCH_4A: f32 = 440.0;
const OCTAVE_RESOLUTION: i32 = 12 * 2;
const OCTAVE_COUNT: i32 = 8;
const NOTE_COUNT: i32 = OCTAVE_RESOLUTION * OCTAVE_COUNT;
const SAMPLE_RATE: f32 = 44100.0;
const FRAME_RATE: f32 = 60.0;
const PITCH_STEP: f32 = SAMPLE_RATE / (FFT_SIZE as f32);

const IMAGE_WIDTH: u32 = 1600;
const IMAGE_HEIGHT: u32 = 1200;

const FRACTAL_SPACE_WIDTH: u32 = 480;
const FRACTAL_SPACE_HEIGHT: u32 = 360;
//const IMAGE_WIDTH: u32 = 1920;
//const IMAGE_HEIGHT: u32 = 1080;

// 2 ^ 24 / 2 - 1
const I24_MAX: u32 = 8388608 - 1;

pub trait SceneWriter {
    fn write(&mut self, video_writer: &mut VideoWriter);
}

pub struct Scene0 {
    splash_at: f32,
    end_at: f32,
    rng: ThreadRng,
    particle_speed: f32,
}

pub struct Scene1 {
    boot_time: f32,
    start_at: f32,
    end_at: f32,
    start_dot_mask_at: f32,
    end_dot_mask_at: f32,
    gain_floor: f32,
    headroom: f32,
    rng: ThreadRng,
    particles: Vec<Particle>,
    paritcle_max_per_pitch: i32,
    velocity_base: f32,
    particle_ttl_base: f32,
}

pub struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
    radius: f32,
    ttl: f32,
    color: Scalar,
}

pub struct MyFractal {
    base_length: f32,
    polygons: VectorOfPoint,
    polygons_f: VectorOfPoint2f,
    color: Scalar,
    theta_factor: f32,
    time: f32,
    curvature: f32,
}

impl MyFractal {
    pub fn child(&self) -> MyFractal {
        let polygon_org = &self.polygons_f;

        let origin = polygon_org.get(0).unwrap();
        let origin = Point2f::new(origin.x as f32, origin.y as f32);

        let dir1 = polygon_org.get(1).unwrap();
        let dir1 = Point2f::new(dir1.x as f32, dir1.y as f32);
        let dir1 = Point2f::new(dir1.x - origin.x, dir1.y - origin.y);
        let dir1_len = dir1.norm() as f32;
        let dir1_theta = (dir1.y as f32).atan2(dir1.x as f32);
        let dir1_theta = dir1_theta + (1.2 + (self.time / 10.0).sin()) / 3.0 * self.theta_factor;
        let mut dir1 = Point2f::new(
            dir1_theta.cos() * self.base_length,
            dir1_theta.sin() * self.base_length,
        );

        let dir2 = polygon_org.get(2).unwrap();
        let dir2 = Point2f::new(dir2.x as f32, dir2.y as f32);
        let dir2 = Point2f::new(dir2.x - origin.x, dir2.y - origin.y);
        let dir2_len = dir2.norm() as f32;
        let dir2_theta = (dir2.y as f32).atan2(dir2.x as f32);
        let dir2_theta = dir2_theta + (1.2 + (self.time / 10.0).cos()) / 3.0 * self.theta_factor;
        let mut dir2 = Point2f::new(dir2_theta.cos() * dir2_len, dir2_theta.sin() * dir2_len);

        if dir1.x.is_nan() {
            dir1.x = 0.0;
        }
        if dir1.y.is_nan() {
            dir1.y = 0.0;
        }
        if dir2.x.is_nan() {
            dir2.x = 0.0;
        }
        if dir2.y.is_nan() {
            dir2.y = 0.0;
        }

        let polygon = VectorOfPoint::from_iter(vec![
            Point2i::new((origin.x + dir1.x) as i32, (origin.y + dir1.y) as i32),
            Point2i::new(
                (origin.x + dir1.x * 1.8 / self.curvature) as i32,
                (origin.y + dir1.y * 1.8 / self.curvature) as i32,
            ),
            Point2i::new(
                (origin.x + dir1.x + dir2.x) as i32,
                (origin.y + dir1.y + dir2.y) as i32,
            ),
        ]);

        let polygon_f = VectorOfPoint2f::from_iter(vec![
            Point2f::new((origin.x + dir1.x) as f32, (origin.y + dir1.y) as f32),
            Point2f::new(
                (origin.x + dir1.x * 1.8 / self.curvature) as f32,
                (origin.y + dir1.y * 1.8 / self.curvature) as f32,
            ),
            Point2f::new(
                (origin.x + dir1.x + dir2.x) as f32,
                (origin.y + dir1.y + dir2.y) as f32,
            ),
        ]);

        let rgb = color_space::Rgb::new(self.color[0], self.color[1], self.color[2]);
        let mut hsv = color_space::Hsv::from(rgb);
        hsv.h = (hsv.h + 1.1).rem_euclid(256.0);
        let rgb = color_space::Rgb::from(hsv);
        let color = Scalar::new(rgb.r, rgb.g, rgb.b, 1.0);
        MyFractal {
            polygons: polygon,
            polygons_f: polygon_f,
            color: color,
            base_length: self.base_length,
            curvature: self.curvature + 0.0002,
            time: self.time,
            theta_factor: self.theta_factor - 0.01,
        }
    }
}

impl SceneWriter for Scene0 {
    fn write(&mut self, video_writer: &mut VideoWriter) {
        let mut particles = Vec::new();

        let mut mat = Mat::zeros(IMAGE_HEIGHT as i32, IMAGE_WIDTH as i32, CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();

        imgproc::rectangle(
            &mut mat,
            Rect {
                x: 0,
                y: 0,
                width: IMAGE_WIDTH as i32,
                height: IMAGE_HEIGHT as i32,
            },
            Scalar::new(255.0, 255.0, 255.0, 1.0),
            -1,
            LineTypes::LINE_4 as i32,
            0,
        )
        .unwrap();

        imgproc::put_text(
            &mut mat,
            "Open CV",
            Point::new(IMAGE_WIDTH as i32 / 2, IMAGE_HEIGHT as i32 / 2),
            2,
            4.0,
            Scalar::new(0.0, 0.0, 0.0, 1.0),
            10,
            0,
            false,
        )
        .unwrap();

        for y in 0..IMAGE_HEIGHT as i32 {
            for x in 0..IMAGE_WIDTH as i32 {
                let p = mat.at_2d::<Vec3b>(y, x).unwrap();
                if *p == Vec3b::from([0, 0, 0]) {
                    if self.rng.gen_ratio(1, 2) {
                        let color = Scalar::new(
                            self.rng.gen_range(
                                220.0 - 100.0 * (x as f64 / IMAGE_WIDTH as f64),
                                335.0 - 100.0 * (x as f64 / IMAGE_WIDTH as f64),
                            ),
                            self.rng.gen_range(180.0, 235.0),
                            self.rng.gen_range(
                                30.0 + 150.0 * (x as f64 / IMAGE_WIDTH as f64),
                                125.0 + 180.0 * (x as f64 / IMAGE_WIDTH as f64),
                            ),
                            1.0,
                        );

                        particles.push(Particle {
                            position: [x as f32, y as f32],
                            velocity: [
                                self.rng.gen::<f32>() * 2.0 * self.particle_speed
                                    - self.particle_speed,
                                self.rng.gen::<f32>() * 2.0 * self.particle_speed
                                    - self.particle_speed,
                            ],
                            radius: self.rng.gen_range(0.3, 3.0),
                            ttl: self.end_at - self.splash_at,
                            color: color,
                        })
                    }
                }
            }
        }

        for idx in 0..(FRAME_RATE * self.end_at) as u32 {
            let mut mat = Mat::zeros(IMAGE_HEIGHT as i32, IMAGE_WIDTH as i32, CV_8UC3)
                .unwrap()
                .to_mat()
                .unwrap();

            imgproc::rectangle(
                &mut mat,
                Rect {
                    x: 0,
                    y: 0,
                    width: IMAGE_WIDTH as i32,
                    height: IMAGE_HEIGHT as i32,
                },
                Scalar::new(240.0, 240.0, 240.0, 1.0),
                -1,
                LineTypes::LINE_4 as i32,
                0,
            )
            .unwrap();

            if self.splash_at < idx as f32 / FRAME_RATE as f32 {
                particles.iter_mut().for_each(|p| {
                    p.position[0] += p.velocity[0] / FRAME_RATE;
                    p.position[1] += p.velocity[1] / FRAME_RATE;
                    p.ttl -= 1.0 / FRAME_RATE;
                    p.velocity[0] *= 0.9999999;
                    p.velocity[1] *= 0.9999999;
                });

                particles.retain(|p| 0.0 < p.ttl);
            }

            particles.iter().for_each(|p| {
                imgproc::circle(
                    &mut mat,
                    Point2i {
                        x: p.position[0] as i32,
                        y: p.position[1] as i32,
                    },
                    p.radius as i32,
                    p.color,
                    -1,
                    LineTypes::LINE_AA as i32,
                    0,
                )
                .unwrap();
            });

            video_writer.write(&mat).unwrap();
        }

        /*
        self.particles.iter_mut().for_each(|p| {
                    p.position[0] += p.velocity[0] / FRAME_RATE;
                    p.position[1] += p.velocity[1] / FRAME_RATE;
                    p.ttl -= 1.0 / FRAME_RATE;
                });

                self.particles.retain(|p| 0.0 < p.ttl);

                */
    }
}

impl SceneWriter for Scene1 {
    fn write(&mut self, video_writer: &mut VideoWriter) {
        let wav_reader = hound::WavReader::open("resource/music.wav").unwrap();
        let wav_spec = wav_reader.spec();
        assert_eq!(SAMPLE_RATE, wav_spec.sample_rate as f32);
        assert_eq!(wav_spec.channels, 2);
        let pitches = (0..FFT_SIZE / 2).map(|i| i as f32 * PITCH_STEP);

        // a4 - 4 octave ..= a4 + 4octave
        // v[0] a0 v[1] b0 v[2] c0 ...
        let pitch_separations = (-OCTAVE_RESOLUTION * OCTAVE_COUNT / 2
            ..=OCTAVE_RESOLUTION * OCTAVE_COUNT / 2)
            .map(|i| PITCH_4A * 2.0_f32.powf((i as f32 - 0.5) / OCTAVE_RESOLUTION as f32));

        let pitch_separation_indices =
            pitch_separations.map(|ps| pitches.clone().position(|p| ps <= p).unwrap() - 1);

        // [ (l, r), (l, r), (l, r), ... ]
        let pitch_separation_indices = pitch_separation_indices
            .clone()
            .take(pitch_separation_indices.clone().count() - 1)
            .zip(pitch_separation_indices.skip(1))
            .map(|(il, ir)| {
                //assert!(il < ir);
                (il, ir)
            })
            .collect::<Vec<(usize, usize)>>();

        let samples = {
            match (wav_spec.sample_format, wav_spec.bits_per_sample) {
                (hound::SampleFormat::Float, 32) => wav_reader
                    .into_samples::<f32>()
                    .map(|v| v.unwrap())
                    .map(|v| Complex32::new(v, 0.0))
                    .collect::<Vec<Complex32>>(),
                (hound::SampleFormat::Int, 16) => wav_reader
                    .into_samples::<i32>()
                    .map(|v| (v.unwrap() as f32) / (std::i16::MAX as f32))
                    .map(|v| Complex32::new(v, 0.0))
                    .collect::<Vec<Complex32>>(),
                (hound::SampleFormat::Int, 24) => wav_reader
                    .into_samples::<i32>()
                    .map(|v| (v.unwrap() as f32) / (I24_MAX as f32))
                    .map(|v| Complex32::new(v, 0.0))
                    .collect::<Vec<Complex32>>(),
                _ => unimplemented!("{:?}, {}", wav_spec.sample_format, wav_spec.bits_per_sample),
            }
        };

        let samples = samples
            .into_iter()
            // cuz its stereo, times 2
            .skip((2.0 * SAMPLE_RATE * self.start_at) as usize)
            .take((2.0 * SAMPLE_RATE * self.end_at) as usize);

        let mut samples = samples
            .clone()
            .step_by(2)
            .zip(samples.skip(1).step_by(2))
            .collect::<Vec<(Complex32, Complex32)>>();

        let mut fft_planner = rustfft::FFTplanner::new(false);
        let fft = fft_planner.plan_fft(FFT_SIZE);
        let mut fractal_theta_base = 0.0;
        let mut time_for_fractal = 0.0;

        let mut mat_dot_mask = Mat::zeros(IMAGE_HEIGHT as i32, IMAGE_WIDTH as i32, CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();

        for i in 0..(IMAGE_WIDTH * IMAGE_HEIGHT) as i32 {
            let p = mat_dot_mask.at_mut(i).unwrap();
            if self.rng.gen_ratio(1, 6) {
                *p = Vec3b::from([255, 255, 255]);
            }
        }
        let kernel = imgproc::get_structuring_element(
            imgproc::MorphShapes::MORPH_ELLIPSE as i32,
            Size::new(5, 5),
            Point::new(0, 0),
        )
        .unwrap();

        imgproc::dilate(
            &mut mat_dot_mask.clone(),
            &mut mat_dot_mask,
            &kernel,
            Point::new(0, 0),
            1,
            opencv::core::BorderTypes::BORDER_CONSTANT as i32,
            Scalar::new(0.0, 0.0, 0.0, 1.0),
        )
        .unwrap();

        for idx in (0..samples.len() - FFT_SIZE).step_by((SAMPLE_RATE / FRAME_RATE) as usize) {
            let time_elapsed = idx as f32 / SAMPLE_RATE;
            let boot_state = (self.boot_time - time_elapsed).max(0.0) * self.gain_floor;
            let input = &mut samples[idx..idx + FFT_SIZE];

            // apply window
            let input = input
                .iter()
                .zip(apodize::hanning_iter(FFT_SIZE))
                .map(|(&v, w)| [v.0 * (w as f32), v.1 * (w as f32)]);

            let mut input_l = input.clone().map(|[l, _r]| l).collect::<Vec<Complex32>>();
            let mut input_r = input.map(|[_l, r]| r).collect::<Vec<Complex32>>();

            let mut output_l = [Complex32::new(0.0, 0.0); FFT_SIZE];
            let mut output_r = [Complex32::new(0.0, 0.0); FFT_SIZE];

            // execute fft
            fft.process(&mut input_l.as_mut_slice(), &mut output_l);
            fft.process(&mut input_r.as_mut_slice(), &mut output_r);

            // cut at nyquist frequency
            let output_l = output_l
                .iter()
                .copied()
                .take(FFT_SIZE / 2)
                .collect::<Vec<Complex32>>();

            let output_r = output_l
                .iter()
                .copied()
                .take(FFT_SIZE / 2)
                .collect::<Vec<Complex32>>();

            // normalize
            let output_l = output_l
                .iter()
                .map(|v| v.norm())
                .map(|v| v / (FFT_SIZE as f32))
                .collect::<Vec<f32>>();

            let output_r = output_r
                .iter()
                .map(|v| v.norm())
                .map(|v| v / (FFT_SIZE as f32))
                .collect::<Vec<f32>>();

            let lin_per_pitch_l = pitch_separation_indices.iter().map(|(il, ir)| {
                let mut n = ir - il;
                if n == 0 {
                    n = 1;
                }
                let ic = il + n / 2;
                output_l.iter().skip(*il).take(n).cloned().sum::<f32>() / (n as f32)
            });

            let lin_per_pitch_r = pitch_separation_indices.iter().map(|(il, ir)| {
                let mut n = ir - il;
                if n == 0 {
                    n = 1;
                }
                let ic = il + n / 2;
                output_r.iter().skip(*il).take(n).cloned().sum::<f32>() / (n as f32)
            });

            // convert to decibel
            let db_per_pitch_l = lin_per_pitch_l
                .clone()
                //.into_iter()
                .map(|v| 20.0 * (v / GAIN_0).log10())
                .enumerate()
                .map(|(i, v)| {
                    (
                        PITCH_4A
                            * 2.0_f32
                                .powf((i as f32 - NOTE_COUNT as f32) / OCTAVE_RESOLUTION as f32),
                        v,
                    )
                })
                .map(|(f, v)| util::loudness_a_weighting(f) + v)
                .collect::<Vec<f32>>();

            let db_per_pitch_r = lin_per_pitch_r
                .clone()
                //.into_iter()
                .map(|v| 20.0 * (v / GAIN_0).log10())
                .enumerate()
                .map(|(i, v)| {
                    (
                        PITCH_4A
                            * 2.0_f32
                                .powf((i as f32 - NOTE_COUNT as f32) / OCTAVE_RESOLUTION as f32),
                        v,
                    )
                })
                .map(|(f, v)| util::loudness_a_weighting(f) + v)
                .collect::<Vec<f32>>();

            let gain = {
                let sqaured_sum = lin_per_pitch_l
                    .clone()
                    .chain(lin_per_pitch_r)
                    .map(|lin| lin.powi(2))
                    .sum::<f32>();
                (sqaured_sum / lin_per_pitch_l.len() as f32 / 2.0).sqrt()
            };

            // -96 dB
            let gain = (gain - 0.00001584893).max(0.0);

            time_for_fractal += 0.8 * gain;
            fractal_theta_base += 0.9 * gain;

            let mut mat_fractal_back = Mat::zeros(
                FRACTAL_SPACE_HEIGHT as i32,
                FRACTAL_SPACE_WIDTH as i32,
                CV_8UC3,
            )
            .unwrap()
            .to_mat()
            .unwrap();

            let mut mat_fractal_front = Mat::zeros(
                FRACTAL_SPACE_HEIGHT as i32,
                FRACTAL_SPACE_WIDTH as i32,
                CV_8UC3,
            )
            .unwrap()
            .to_mat()
            .unwrap();

            let mut mat_fractal_front2 = Mat::zeros(
                FRACTAL_SPACE_HEIGHT as i32,
                FRACTAL_SPACE_WIDTH as i32,
                CV_8UC3,
            )
            .unwrap()
            .to_mat()
            .unwrap();

            let mut mat_fractal_blended = Mat::zeros(
                FRACTAL_SPACE_HEIGHT as i32,
                FRACTAL_SPACE_WIDTH as i32,
                CV_8UC3,
            )
            .unwrap()
            .to_mat()
            .unwrap();

            let base_color = color_space::Hsv::new(
                (time_for_fractal as f64 * 2.2 + 150.0).rem_euclid(256.0),
                gain as f64 * 500.0 - 0.25,
                (time_for_fractal as f64 * 1.14).sin() / 8.0 + (1.0 / 8.0),
            );
            let base_color = color_space::Rgb::from(base_color);
            let base_color = Scalar::new(
                base_color.r * 0.9,
                base_color.g * 0.9,
                base_color.b * 0.9,
                1.0,
            );

            imgproc::rectangle(
                &mut mat_fractal_back,
                Rect {
                    x: 0,
                    y: 0,
                    width: FRACTAL_SPACE_WIDTH as i32,
                    height: FRACTAL_SPACE_HEIGHT as i32,
                },
                //Scalar::new(203.0, 186.0, 74.0, 1.0),
                base_color,
                -1,
                LineTypes::LINE_4 as i32,
                0,
            )
            .unwrap();

            imgproc::rectangle(
                &mut mat_fractal_front,
                Rect {
                    x: 0,
                    y: 0,
                    width: FRACTAL_SPACE_WIDTH as i32,
                    height: FRACTAL_SPACE_HEIGHT as i32,
                },
                //Scalar::new(203.0, 186.0, 74.0, 1.0),
                Scalar::new(0.0, 0.0, 0.0, 0.0),
                -1,
                LineTypes::LINE_4 as i32,
                0,
            )
            .unwrap();

            imgproc::rectangle(
                &mut mat_fractal_front2,
                Rect {
                    x: 0,
                    y: 0,
                    width: FRACTAL_SPACE_WIDTH as i32,
                    height: FRACTAL_SPACE_HEIGHT as i32,
                },
                //Scalar::new(203.0, 186.0, 74.0, 1.0),
                Scalar::new(190.0, 190.0, 190.0, 1.0),
                -1,
                LineTypes::LINE_4 as i32,
                0,
            )
            .unwrap();

            let mut fractals = Vec::new();

            // fractal back
            for theta in (0..360).step_by(15) {
                let theta = fractal_theta_base + theta as f32 / 360.0 * std::f32::consts::TAU;
                let theta2 = theta - 0.2 * ((time_for_fractal / 9.0).cos() / 2.0 + 0.5);

                let base_color = color_space::Hsv::new(
                    (time_for_fractal as f64 * 39.2 + 100.0).rem_euclid(256.0),
                    time_for_fractal.cos() as f64 / 3.0 + 0.75,
                    1.2 - gain as f64 * 70.0,
                );
                let base_color = color_space::Rgb::from(base_color);
                let base_color = Scalar::new(base_color.r, base_color.g, base_color.b, 1.0);

                let my_fractal = MyFractal {
                    polygons: VectorOfPoint::from_iter(vec![
                        Point2i::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0) as i32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0) as i32,
                        ),
                        Point2i::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0
                                + 40.0 * theta.cos() * (time_for_fractal / 81.4).cos())
                                as i32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0
                                + 40.0 * theta.sin() * (time_for_fractal / 81.4).cos())
                                as i32,
                        ),
                        Point2i::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0
                                + 173.0 * (theta2.cos() * (time_for_fractal / 17.4).cos()))
                                as i32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0
                                + 173.0 * (theta2.sin() * (time_for_fractal / 17.4).cos()))
                                as i32,
                        ),
                    ]),
                    polygons_f: VectorOfPoint2f::from_iter(vec![
                        Point2f::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0) as f32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0) as f32,
                        ),
                        Point2f::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0
                                + 40.0 * theta.cos() * (time_for_fractal / 81.4).cos())
                                as f32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0
                                + 40.0 * theta.sin() * (time_for_fractal / 81.4).cos())
                                as f32,
                        ),
                        Point2f::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0
                                + 173.0 * (theta2.cos() * (time_for_fractal / 17.4).cos()))
                                as f32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0
                                + 173.0 * (theta2.sin() * (time_for_fractal / 17.4).cos()))
                                as f32,
                        ),
                    ]),
                    color: base_color,
                    base_length: 23.0,
                    time: time_for_fractal,
                    curvature: 1.0,
                    theta_factor: (time_for_fractal / 20.2).cos() / 5.0 + 1.2,
                };

                fractals.push(my_fractal);
            }

            for _ in 0..128 {
                fractals.iter_mut().for_each(|f| {
                    imgproc::fill_convex_poly(
                        &mut mat_fractal_back,
                        &f.polygons as &dyn ToInputArray,
                        f.color,
                        LineTypes::LINE_AA as i32,
                        0,
                    )
                    .unwrap();

                    *f = f.child();
                });
            }

            // fractal front
            for theta in (0..360).step_by(30) {
                let theta = fractal_theta_base + theta as f32 / 360.0 * std::f32::consts::TAU;
                let theta2 = theta - 0.2 * ((time_for_fractal / 5.0).cos() / 2.0 + 0.5);

                let base_color = color_space::Hsv::new(
                    (time_for_fractal as f64 * 27.8).rem_euclid(256.0),
                    time_for_fractal.sin() as f64 / 3.0 + 0.75,
                    0.4 + gain as f64 * 75.0,
                );
                let base_color = color_space::Rgb::from(base_color);
                let base_color = Scalar::new(base_color.r, base_color.g, base_color.b, 1.0);

                let my_fractal = MyFractal {
                    polygons: VectorOfPoint::from_iter(vec![
                        Point2i::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0) as i32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0) as i32,
                        ),
                        Point2i::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0
                                + 20.0 * theta.cos() * (time_for_fractal / 11.4).cos())
                                as i32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0
                                + 20.0 * theta.sin() * (time_for_fractal / 11.4).cos())
                                as i32,
                        ),
                        Point2i::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0
                                + 73.0 * theta2.cos() * (time_for_fractal / 14.4).cos())
                                as i32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0
                                + 73.0 * theta2.sin() * (time_for_fractal / 14.4).cos())
                                as i32,
                        ),
                    ]),
                    polygons_f: VectorOfPoint2f::from_iter(vec![
                        Point2f::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0) as f32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0) as f32,
                        ),
                        Point2f::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0
                                + 20.0 * theta.cos() * (time_for_fractal / 11.4).cos())
                                as f32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0
                                + 20.0 * theta.sin() * (time_for_fractal / 11.4).cos())
                                as f32,
                        ),
                        Point2f::new(
                            (FRACTAL_SPACE_WIDTH as f32 / 2.0
                                + 73.0 * theta2.cos() * (time_for_fractal / 14.4).cos())
                                as f32,
                            (FRACTAL_SPACE_HEIGHT as f32 / 2.0
                                + 73.0 * theta2.sin() * (time_for_fractal / 14.4).cos())
                                as f32,
                        ),
                    ]),
                    color: base_color,
                    base_length: 22.0,
                    time: time_for_fractal,
                    curvature: 1.0,
                    theta_factor: (time_for_fractal / 7.2).cos() / 3.0 + 1.2,
                };

                fractals.push(my_fractal);
            }

            for _ in 0..128 {
                fractals.iter_mut().for_each(|f| {
                    imgproc::fill_convex_poly(
                        &mut mat_fractal_front,
                        &f.polygons as &dyn ToInputArray,
                        f.color,
                        LineTypes::LINE_AA as i32,
                        0,
                    )
                    .unwrap();

                    *f = f.child();
                });
            }

            imgproc::gaussian_blur(
                &mut mat_fractal_back.clone(),
                &mut mat_fractal_back,
                Size::new(21, 21),
                20.0,
                0.0,
                BORDER_REFLECT as i32,
            )
            .unwrap();

            opencv::core::add_weighted(
                &mut mat_fractal_back,
                0.6,
                &mut mat_fractal_front,
                0.6,
                0.0,
                &mut mat_fractal_blended,
                -1,
            )
            .unwrap();

            let mut mat_fractal_expanded =
                Mat::zeros(IMAGE_HEIGHT as i32, IMAGE_WIDTH as i32, CV_8UC3)
                    .unwrap()
                    .to_mat()
                    .unwrap();

            let size = mat_fractal_expanded.size().unwrap();
            imgproc::resize(
                &mat_fractal_blended,
                &mut mat_fractal_expanded,
                size,
                0.0,
                0.0,
                opencv::imgproc::InterpolationFlags::INTER_NEAREST as i32,
            )
            .unwrap();

            let mut mat_fractal_expanded = mat_fractal_expanded;

            let mut mat_spectrum = Mat::zeros(IMAGE_HEIGHT as i32, IMAGE_WIDTH as i32, CV_8UC3)
                .unwrap()
                .to_mat()
                .unwrap();

            for (pitch_idx, (db_l, db_r)) in
                db_per_pitch_l.iter().zip(db_per_pitch_r.iter()).enumerate()
            {
                let db_l = db_l + 1.5 * boot_state;
                let db_r = db_r + 1.5 * boot_state;

                const HEIGHT_PER_PITCH: f32 = IMAGE_HEIGHT as f32 / NOTE_COUNT as f32;
                let y_max = IMAGE_HEIGHT as f32 - ((pitch_idx - 1) as f32 * HEIGHT_PER_PITCH);
                let y_min = IMAGE_HEIGHT as f32 - ((pitch_idx + 1 + 1) as f32 * HEIGHT_PER_PITCH);

                let bar_color = Scalar::new(
                    100.0 + 2.0 * boot_state as f64,
                    100.0 + 2.0 * boot_state as f64,
                    100.0 + 2.0 * boot_state as f64,
                    1.0,
                );
                imgproc::rectangle(
                    &mut mat_spectrum,
                    Rect {
                        x: 0,
                        y: y_min as i32,
                        width: (IMAGE_WIDTH as f32 * (1.0 + db_l / self.gain_floor) / 2.0) as i32,
                        height: (y_max - y_min) as i32,
                    },
                    bar_color,
                    -1,
                    LineTypes::LINE_4 as i32,
                    0,
                )
                .unwrap();

                imgproc::rectangle(
                    &mut mat_spectrum,
                    Rect {
                        x: IMAGE_WIDTH as i32
                            - 1
                            - (IMAGE_WIDTH as f32 * (1.0 + db_r / self.gain_floor) / 2.0) as i32,
                        y: y_min as i32,
                        width: (IMAGE_WIDTH as f32 * (1.0 + db_r / self.gain_floor) / 2.0) as i32,
                        height: (y_max - y_min) as i32,
                    },
                    bar_color,
                    -1,
                    0,
                    0,
                )
                .unwrap();

                let mut mat_blended = Mat::zeros(IMAGE_HEIGHT as i32, IMAGE_WIDTH as i32, CV_8UC3)
                    .unwrap()
                    .to_mat()
                    .unwrap();

                let mut mat_out = Mat::zeros(IMAGE_HEIGHT as i32, IMAGE_WIDTH as i32, CV_8UC3)
                    .unwrap()
                    .to_mat()
                    .unwrap();

                opencv::core::bitwise_or(
                    &mut mat_spectrum,
                    &mut mat_fractal_expanded,
                    &mut mat_out,
                    &mut no_array().unwrap(),
                )
                .unwrap();

                let mut mat_map = Mat::zeros(IMAGE_HEIGHT as i32, IMAGE_WIDTH as i32, CV_32FC2)
                    .unwrap()
                    .to_mat()
                    .unwrap();

                video_writer.write(&mat_out).unwrap();
            }
        }
    }
}

fn main() -> std::io::Result<()> {
    dbg!(PITCH_STEP);

    // https://www.fourcc.org/codecs.php
    // FFMpeg
    let mut video_writer = VideoWriter::new(
        "test.avi",
        VideoWriter::fourcc('H' as i8, '2' as i8, '6' as i8, '4' as i8).unwrap(),
        60.0,
        opencv::core::Size::new(IMAGE_WIDTH as i32, IMAGE_HEIGHT as i32),
        true,
    )
    .unwrap();

    assert!(video_writer.is_opened().unwrap());

    let rng = rand::thread_rng();
    let mut scene0 = Scene0 {
        splash_at: 8.2,
        end_at: 12.0,
        rng: rng,
        particle_speed: 2000.0,
    };
    scene0.write(&mut video_writer);

    let rng = rand::thread_rng();
    let mut scene1 = Scene1 {
        boot_time: 1.0,
        start_at: scene0.end_at,
        end_at: 55.0,
        //end_at: 15.0,
        start_dot_mask_at: 12.0,
        end_dot_mask_at: 20.0,
        gain_floor: 96.0,
        headroom: 6.0,
        rng: rng,
        particles: Vec::new(),
        paritcle_max_per_pitch: 3,
        velocity_base: 3.0,
        particle_ttl_base: 300.0,
    };
    scene1.write(&mut video_writer);

    video_writer.release().unwrap();

    Ok(())
}

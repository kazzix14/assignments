pub fn loudness_a_weighting(freq: f32) -> f32 {
    let freqs = [
        6.0, 3.0, 8.0, 10.0, 12.0, 5.0, 16.0, 20.0, 25.0, 31.0, 5.0, 40.0, 50.0, 63.0, 80.0, 100.0,
        125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
        2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
    ];
    let a = [
        0.00047573,
        0.00047573,
        0.00430438,
        0.0147611,
        -0.00289718,
        0.00164277,
        0.00054718,
        0.00053251,
        -0.00009334,
        0.00010867,
        0.00003418,
        -0.00000419,
        0.00000565,
        0.00000722,
        -0.00000153,
        0.00000065,
        0.00000036,
        0.00000017,
        -0.00000005,
        0.00000006,
        0.0,
        0.00000001,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
    ];
    let b = [
        -0.21083602,
        -0.20840981,
        -0.20555544,
        -0.17327257,
        -0.01828098,
        -0.05304711,
        -0.02840562,
        -0.0177356,
        -0.00415666,
        -0.00695686,
        -0.00271855,
        -0.00097552,
        -0.00122688,
        -0.00080283,
        -0.00004457,
        -0.00022859,
        -0.00013142,
        -0.0000621,
        -0.00001889,
        -0.00003261,
        -0.00000854,
        -0.00000722,
        -0.00000309,
        -0.0000023,
        -0.00000069,
        -0.00000023,
        -0.00000044,
        0.00000009,
        -0.00000026,
        0.0000001,
        -0.00000008,
        -0.00000003,
        -0.00000001,
        0.00000002,
        -0.0,
    ];
    let c = [
        4.82763462,
        4.11491671,
        3.28698621,
        2.33991619,
        1.66947876,
        1.3841664,
        0.97690276,
        0.67698484,
        0.49090059,
        0.37976543,
        0.25398519,
        0.19118609,
        0.14713819,
        0.09639555,
        0.06673648,
        0.05580997,
        0.03780956,
        0.02523078,
        0.01834645,
        0.0131962,
        0.00784708,
        0.00516867,
        0.0031066,
        0.00175947,
        0.0007139,
        0.00034799,
        0.00001666,
        -0.00021054,
        -0.00035947,
        -0.00051801,
        -0.00048447,
        -0.00066193,
        -0.00072482,
        -0.0006959,
        -0.00064036,
    ];
    let d = [
        -85.4, -77.8, -70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1,
        -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0,
        0.5, -0.1, -1.1, -2.5, -4.3, -6.6,
    ];

    let idx = freqs
        .iter()
        .position(|&f| freq < f)
        .unwrap_or(freqs.len() - 1)
        .max(1)
        .min(freqs.len() - 1)
        - 1;

    let freq_diff = freq - freqs[idx];
    let weighting_intensity = 0.2;

    weighting_intensity
        * (d[idx] + c[idx] * freq_diff + b[idx] * freq_diff.powi(2) + a[idx] * freq_diff.powi(3))
}

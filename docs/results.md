# Measurements and media

[Русский](results.ru.md) · [Home](../README.md) · [Gallery](../assets/results/index.html)

Recorded 26 October 2026 on Pi 5 + Camera Module 3 Standard/IMX708. [Measurements and versions](../assets/results/measurements.json). Full indoor captures are not redistributed. Aggregates reproduce charts, not independent raw-frame analysis. The reorganized package has not been rerun on hardware.

## Motion

![Line width](../assets/results/line-width.png)

One sequential trial per mode used a hand-moved contrast card. Affine registration aligned a dark feature; apparent horizontal FWHM was compared at 150–250 reference-pixel/s. Rejected fits and adjacent sensor gaps >50 ms were excluded.

| Mode | Median FWHM | Frame IQR | Eligible frames |
| --- | --- | --- | --- |
| Native Short | 5.50 px | 5–6 | 117 |
| Native Normal | 6.375 px | 5.5–6.75 | 100 |
| Fixed 4 ms | 3.00 px | 2.75–4 | 60 |
| Adaptive | 4.00 px | 3.75–4.5 | 69 |

Adaptive lines were 27.3%/37.3% narrower than native Short/Normal. Fixed 4 ms was sharper in motion. Native exposure was actually 32.680 ms in both profiles: Short naming did not enforce a short shutter in this light. Fixed read back 3.981 ms; adaptive moving exposure 2.779–7.989 ms. After stopping, adaptive restored native 32.680 ms. Fixed gain reached 16; restored native gain was approximately 9.39. This is an observed trade-off, not a calibrated SNR improvement.

Adaptive validation: 436 frames, 270 accepted affine fits, 264 velocity-eligible frames, 69 in the comparison band. All 15 new commands were observed in metadata; 18/18 in the broader summary includes three setup commands. Maximum sensor gap: 266.6 ms. Nominal 30 fps does not imply delivered analysis/preview throughput.

FWHM includes target thickness, optics, ISP and resampling, not pure motion blur. Adjacent frames are correlated; rejection rates differ; trial order was not randomized; hand motion/light may vary. Analysis is exploratory. No significance, independent-repeat confidence interval, detector recall or road-object-size claim is established. Chart whiskers show frame quartiles.

## Video

[Play MP4](../assets/results/target-motion.mp4) · [Provenance](../assets/results/video-provenance.json)

Five seconds, four panels: real saved 8-bit luminance, cropped to the inner card and registered to hide surroundings. Panels are separate recordings aligned by motion onset, not simultaneous or speed-matched frame by frame. Overlays show actual exposure/speed. Rejected fits are black. No brightness normalization or fabricated intermediate frames. Poster is illustrative; chart uses the common speed band. Live preview is colour; historical measurement footage is luminance.

## Resources

![CPU](../assets/results/cpu-cost.png)

Isolated tracker/policy with preloaded data: 84 calls at 0.12 s intervals, 10.080075 s wall and 1.652639 s CPU: 16.4% of one core / 4.1% of four-core Pi 5. An unpaced 505-call run measured mean 15.542 ms, median 15.243 ms, p95 21.329 ms, maximum 23.706 ms. This is not worst-case execution time.

Stream-like isolated process: approximately 28–29 MiB with Python/NumPy, 5–6 MiB above its 23 MiB baseline. Preloading 101 replay frames added approximately 18 MiB for that dataset, not streaming algorithm state. Complete colour/web process: RSS 94.97 MiB, peak 96.92 MiB, PSS 86.43 MiB, swap zero. Kernel/DMA/ISP buffers may fall outside RSS. Short measurements do not prove long-term leak freedom.

The isolated algorithm used one OS thread, no video encoder/decoder, no GPU/NPU. CSI/ISP, JPEG and HTTP are additional work excluded from the CPU benchmark. IP-camera decoding would add more work.

## Reproduce and extend

[Development commands](development.md) regenerate aggregate charts. New evidence should preserve configuration, software, requested/actual controls, timestamps and rejection criteria. Repeat randomized native/fixed/adaptive trials at matched speed, light and focus. Report per-trial statistics, contrast/noise and detector accuracy. Road mounting, view, pixel scale and motion need separate measurement. This target test does not validate a two-lane road scene.

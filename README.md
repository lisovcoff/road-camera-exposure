# Road Camera Exposure

[Русский](README.ru.md) · English

Experimental motion-aware shutter control for Raspberry Pi. The controller shortens exposure when a tracked target moves, then restores native auto exposure after it stops. Native gain remains active. Live preview is **colour**; tracking uses luminance.

[![Target comparison](assets/results/demo-poster.png)](assets/results/target-motion.mp4)

[Watch video](assets/results/target-motion.mp4) · [Media gallery](assets/results/index.html) · [Method and limitations](docs/results.md)

## Measured result

![Line width](assets/results/line-width.png)

| Mode | Actual moving exposure | Median line width | Frames at 150–250 px/s |
| --- | --- | --- | --- |
| Native Short | 32.680 ms | 5.50 px | 117 |
| Native Normal | 32.680 ms | 6.375 px | 100 |
| Fixed 4 ms + native gain | 3.981 ms | 3.00 px | 60 |
| Adaptive + native gain | 2.779–7.989 ms | 4.00 px | 69 |

Adaptive control produced a line **27.3% narrower than Native Short and 37.3% narrower than Native Normal** in the common speed band. Fixed 4 ms was sharper during motion. Adaptive restored native 32.680 ms exposure at rest. This was one sequential indoor trial per mode with hand motion and correlated frames. It does not establish statistical significance, road performance or improved detection. Whiskers are frame quartiles, not confidence intervals.

## Included and supported

- Independent NumPy tracker, shutter policy and command/readback gate.
- Picamera2 IMX708 adapter, colour MJPEG crop preview, bounded validation capture.
- Configuration, measured AWB reference, actual metadata, offline analysis and tests.

The original runtime was tested on **Pi 5 + Camera Module 3 Standard / IMX708**. The reorganized package has offline and mocked camera tests; its first real hardware run remains to be checked. Other cameras require an adapter. Road-wide motion, trash detection and automatic road calibration are not implemented.

## Try without a camera

Python 3.10+, commands from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
roadcam simulate
roadcam run --config configs/module3.example.json
python -m unittest discover -s tests -v
```

The `run` command above validates configuration without opening a camera. Simulation checks synthetic feedback, not image quality.

## Raspberry Pi and other cameras

Follow the [installation guide](docs/installation.md), then capture a reference and adjust the generated configuration to your scene:

```sh
roadcam inspect
mkdir -p runs
roadcam reference --config configs/module3.example.json --output runs/reference --run-camera
roadcam run --config runs/reference/config.json
roadcam run --config runs/reference/config.json --output runs/demo-001 --run-camera --allow-lab
```

Default preview: <http://127.0.0.1:8000/> on the Pi. The guide explains access from a trusted LAN; no authentication or TLS is provided. The example target rectangle is illustrative and must be positioned over a textured target.

For another camera, see the [integration guide](docs/camera-adapters.md) and [SDK template](examples/custom_camera.py). CLI currently supports only IMX708. A new backend needs actual controls, unit conversion, timestamps and readback validation. An RTSP URL alone does not supply exposure control.

## Resource cost

![CPU cost](assets/results/cpu-cost.png)

On the measured Pi 5, isolated tracking/policy at 8.3 evaluations/s used **16.4% of one core**, approximately 4.1% of four-core capacity. Mean call: 15.54 ms. Capture, ISP, JPEG and HTTP excluded.

| Scope | Measured cost |
| --- | --- |
| Algorithm with Python/NumPy | 28–29 MiB; approximately 5–6 MiB above baseline |
| Complete colour/web process | 94.97 MiB RSS; 96.92 MiB peak |
| Algorithm alone | One OS thread, zero video encoders/decoders, no GPU/NPU |

Kernel/ISP buffers may fall outside RSS. Short measurements are not worst-case latency or long-term leak tests. [Measurement scope](docs/results.md).

## Documentation

- [Installation](docs/installation.md)
- [Other cameras](docs/camera-adapters.md)
- [Architecture](docs/architecture.md)
- [Results](docs/results.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Development and reproducibility](docs/development.md)
- [Primary references](docs/references.md)

Next validation requires repeated randomized trials, road motion, changing light and detector accuracy measurements. This experimental toolkit is not a safety-critical navigation component.

[MIT license](LICENSE).

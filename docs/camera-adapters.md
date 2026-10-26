# Another camera

[Русский](camera-adapters.ru.md) · [Home](../README.md)

Only the Picamera2 IMX708 runner is implemented. Changing `driver` in JSON does not add sensor support. The generic interface is an integration contract, not a plug-in loader.

Read [base.py](../src/road_camera_exposure/adapters/base.py) and [custom_camera.py](../examples/custom_camera.py). Implement ownership, `capabilities()`, `read()`, `set_shutter()` and `close()` using the manufacturer's SDK. The template intentionally raises `NotImplementedError`.

1. Supply visible `uint8` luminance without stride padding. Keep a separate colour stream for consumers.
2. Normalize exposure to **microseconds**, gain to documented linear factors. V4L2 `EXPOSURE_ABSOLUTE` uses 100 µs units in its specification; check device implementation/bounds. `exposure_to_device_units()` floors quantization; verify actual readback. Other APIs may use different units.
3. Map actual exposure-start timestamps to a monotonic clock. `CapturedFrame.midpoint_ns` adds half the actual exposure. Receipt time is not sensor exposure time; do not invent metadata for IP cameras.
4. Verify manual shutter can coexist with native gain. Otherwise a coupled gain policy needs separate design and testing; this repository supplies no generic replacement.
5. `set_shutter(None)` must restore native shutter. Commands may apply several frames later: use metadata and `ControlGate`, not successful SDK return alone.
6. Connect `TargetTracker` and `ShutterPolicy` in your capture loop. Add a CLI/config backend after SDK integration works. Enforce actual bounds, cleanup and bounded logs.

Another CSI sensor requires mode, focus, metadata and control adaptation. The existing runner checks IMX708 and selects a specific 1080p/2304×1296 mode; do not bypass the check without testing.

For USB/UVC, query available controls and their auto/manual coupling. For RTSP/IP, decoded frames alone cannot change shutter: use a documented vendor/ONVIF control API. Compressed input adds decoder work and latency outside the published algorithm benchmark. Without control/readback, the same closed loop cannot be reproduced.

Validate SDK units → replay → stationary readback → bounded moving-target comparison → representative road trials. Measure command latency, dropped frames, signal/noise, blur, colour, CPU and total latency together. No additional camera has passed these checks here.

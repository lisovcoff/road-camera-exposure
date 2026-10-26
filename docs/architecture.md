# Architecture

[Русский](architecture.ru.md) · [Home](../README.md)

`motion.py` contains the live target tracker, shutter policy and command/readback gate. `adapters/picamera2.py` owns capture/controls; `colour.py` builds colour crops; `web.py` serves preview. `core.py` is a separate tested image-statistics/exposure component used by offline simulation, not the live motion controller. Offline registration and analysis are in `motion_analysis.py` and `analysis/`.

```text
Sensor → ISP → YUV420 → luminance crop → tracker → shutter policy → readback gate
                   └→ colour JPEG → browser                           ↓
                                                               camera shutter
```

The tracker downsamples an explicit target patch and estimates affine motion against its first reference. Speed uses exposure-midpoint timestamps. It measures target image motion, not vehicle speed or road-wide optical flow.

At >=30 px/s the policy selects MOVING; below 10 px/s for one second it selects STILL. Intermediate speeds retain state. Moving shutter uses a heuristic 0.8-pixel blur budget with margin, clipped to 1–8 ms and 15% command hysteresis. Invalid tracking selects a 4 ms laboratory fallback. STILL restores native shutter. Acceleration, weak texture, clipping, command latency and minimum exposure can violate this heuristic. Gain remains native, focus is configured manually, AWB gains measured once.

Commands are logged with actual readback; an SDK return does not mean immediate application. Session time, logs and validation storage are bounded. Algorithm cost excludes camera, ISP, JPEG and HTTP.

Colour preview comes from the same YUV frame as luminance analysis. Historical measurement video uses stored luminance only. The project does not colourize grayscale, perform HDR, run a detector or automatically calibrate camera-to-road geometry.

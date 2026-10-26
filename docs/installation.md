# Raspberry Pi installation

[Русский](installation.ru.md) · [Home](../README.md)

Use Raspberry Pi OS 64-bit and Pi 5 + Camera Module 3 Standard/IMX708. The experiment used Debian 13, Python 3.13.5, Picamera2 0.3.37 and libcamera 0.7.2. These are recorded versions, not a guarantee for every OS release. The reorganized package has offline/mocked tests; verify its first real run on your hardware.

Connect the ribbon with power off. All commands below run **in a Raspberry Pi terminal**, not Windows PowerShell.

## Install

Keep Picamera2 and libcamera together as OS packages; do not replace them with unrelated pip builds. On a minimal image:

```sh
sudo apt update
sudo apt install --no-install-recommends python3-picamera2
sudo apt install git python3-venv python3-pip python3-setuptools
git clone https://github.com/lisovcoff/road-camera-exposure.git
cd road-camera-exposure
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e .
python -c "import picamera2, libcamera, numpy, simplejpeg; print('Camera dependencies available')"
roadcam inspect
```

Expected: dependency message and camera list containing `imx708`. Empty list: check the camera. Other models need an adapter. If missing, install OS packages `python3-numpy` or `python3-simplejpeg`, then retry inside the activated environment.

## Reference and scene setup

Close other camera programs. Place a textured card on the floor; keep camera, card and light still:

```sh
mkdir -p runs
roadcam reference --config configs/module3.example.json --output runs/reference --run-camera
```

Expected: `reference.jpg`, `metadata.json`, `camera.json`, `config.json` and a summary in a **new** directory. This measures native AE/AWB and copies colour gains into configuration. It does not calibrate focus, distance or road geometry.

Open full-resolution `reference.jpg` and edit `runs/reference/config.json`:

| Setting | Meaning |
| --- | --- |
| `roi` | `[x0,y0,x1,y1]` in original **unrotated 1920×1080** image; even coordinates, width 64–640, height 64–360 |
| `tracking_rectangle` | Textured target rectangle relative to crop; at least 32×32, at least 8 pixels inside crop edges |
| `lens` | Manual focus in dioptres, allowed 0–2; inverse metres is only a starting point; inspect detail |
| `colour_gains` | Two measured AWB gains; do not copy another camera's values |
| `rotate_preview` | Rotate displayed crop 180°; coordinates/tracker stay unrotated |
| `bind`, `port` | Default `127.0.0.1`, port 8000 |
| `duration_s` | Session limit, default 3600 seconds |
| storage limits | `session_bytes`, `validation_bytes`, `reserve_bytes`; minimum 1 GB reserve |

After focus changes, recapture the reference if target appearance changes. Each run initializes a fresh tracking reference: keep the target inside the configured rectangle at startup. Example coordinates are illustrative; height and distances are not inferred.

## Run and view

```sh
roadcam run --config runs/reference/config.json
roadcam run --config runs/reference/config.json --output runs/demo-001 --run-camera --allow-lab
```

First: `dry_run: true`, `camera_started: false`. Second: colour MJPEG crop at <http://127.0.0.1:8000/>. `--allow-lab` acknowledges an uncalibrated target demonstration. Stop with Ctrl+C and inspect `summary.json` errors/cleanup. Every run needs a new output directory.

For a phone/laptop on the same trusted LAN:

```sh
hostname -I
```

Choose the Pi's private IPv4, substitute it for the example below:

```sh
roadcam run --config runs/reference/config.json --output runs/demo-002 --run-camera --allow-lab --bind 192.168.1.50
```

Open `http://192.168.1.50:8000/` on the other device. No authentication/TLS: do not forward this port to the Internet. Configure Wi-Fi through Raspberry Pi OS; this project stores no credentials.

The page records one bounded validation: preparation, 3 s still, 6 s moving card, 6 s still. Keep camera/light fixed. Local output includes colour JPEGs, luminance arrays and metadata, excluded from Git. Preview is a crop refreshed around 10 times/s, not a guaranteed full-frame 30 fps transport.

Verify actual exposure/gain, command acknowledgement, timestamps, focus and configuration. Motion should shorten shutter; stopping should restore native shutter. Compare native/fixed/adaptive at matched speeds. In darkness, shorter exposure trades blur for noise.

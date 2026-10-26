# Troubleshooting

[Русский](troubleshooting.ru.md) · [Installation](installation.md)

| Symptom | Check |
| --- | --- |
| Missing Picamera2/libcamera | OS packages, venv with `--system-site-packages`, environment activation |
| No camera / busy | Ribbon with power off; close other capture programs; `roadcam inspect` |
| Unsupported controls | IMX708 and independent shutter/gain modes are required; inspect actual SDK capabilities |
| Missing colour gains | Run `roadcam reference`, use generated `config.json` |
| Output exists | Use a new directory; prior experiments are not overwritten |
| No preview remotely | Pi private IPv4, same LAN and firewall; loopback works only on the Pi |
| Upside-down preview | `rotate_preview`; configured coordinates remain unrotated |
| UNKNOWN / 4 ms | Check ROI, target rectangle, texture, pose and light |
| Dark/noisy image | Short exposure collects fewer photons; add light or reduce motion, compare noise and blur together |
| Session ended | Duration, free space, byte limits, `summary.json`; reserve at least 1 GB |
| Low preview fps | Sensor cadence differs from JPEG/network/browser delivery |

Logs may contain indoor images and local system details. Review before sharing. Preserve originals and write results to new directories.

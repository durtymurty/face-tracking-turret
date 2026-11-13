# Face-Tracking Turret

A pan/tilt turret that finds a face with computer vision and keeps it centered using PID control.

**Status:** 🚧 In progress. Tracking, control, and a virtual turret simulator are working on a laptop webcam. Hardware build is next.

## How it works
1. Camera captures video
2. **YuNet** neural-network face detector (OpenCV) finds faces, including tilted or turned ones
3. Calculates how far the face is from the center of the frame (error, in pixels)
4. **PID controller** turns that error into pan/tilt turning speed
5. Turret moves until the face is centered → **LOCKED**

## Features
- **Robust detection:** YuNet handles head tilts and angles that a classic Haar cascade misses
- **Sticky targeting:** keeps following the same person, even if someone else steps in
- **Smoothing:** filters detection jitter so the motion stays steady
- **Lock-on with hysteresis:** the lock zone scales with face size and gets bigger once locked, so small head movements don't break the lock
- **Virtual turret simulator:** a cropped "turret view" moves around the webcam image like a real camera, so the PID can be tuned before any hardware exists
- **Simulated latency:** models WiFi, detection, and servo delay (150 ms) so the tuning carries over to the real turret
- **Live PID tuning:** sliders adjust KP, KI, and KD while it runs

## Results so far
| Metric | Value |
|---|---|
| Tracking speed (laptop) | ~40–50 FPS |
| Tuned gains (in simulation) | KP = 3.0, KI = 0, KD = 0.06 |

**Tuning notes:** With 150 ms of delay, high KP alone overshoots and oscillates, and KD damps it. Freezing the turret while locked made it lose fast-moving targets, so it now keeps correcting at all times, and "locked" is only a status. KI stays at 0 because the turret's angle already builds up over time, which removes steady-state error on its own.

## Files
| File | What it does |
|---|---|
| `face_track.py` | Face detection + tracking on a fixed webcam |
| `sim_turret.py` | Virtual turret with latency and live PID tuning |
| `face_detection_yunet_2023mar.onnx` | YuNet face detection model (from OpenCV Zoo) |

## Run it
```
pip install "opencv-python<5"
python sim_turret.py
```
Two windows open: **World** (the full scene, with the turret's aim shown as a box) and **Turret view** (what the turret sees, plus PID sliders). Press `q` to quit.

## Hardware (planned)
- Freenove ESP32-S3 camera board
- 2× MG90S metal-gear servos (pan + tilt)
- Active buzzer + LED for lock-on
- 3D-printed pan/tilt mount

## Roadmap
- [x] Face detection + tracking on webcam
- [x] PID control with smoothing and sticky targeting
- [x] Virtual turret simulator with latency + live tuning
- [ ] Scan mode (sweep when no face is visible)
- [ ] Stream ESP32-S3 camera to laptop over WiFi
- [ ] Drive real servos from the tracking output
- [ ] Re-tune PID on real hardware
- [ ] 3D-printed mount + lock-on LED/buzzer
- [ ] Demo video

## Built with
Python · OpenCV · YuNet · PID control · ESP32-S3 · Arduino C++
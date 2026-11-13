# Face-Tracking Turret

A pan/tilt turret that finds a face with computer vision and keeps it centered using PID control with latency compensation.

**Status:** 🚧 In progress. Tracking, control, scan mode, and prediction are working in a virtual turret simulator. Hardware build is next.

## How it works
1. Camera captures video
2. **YuNet** neural-network face detector (OpenCV) finds faces, including tilted or turned ones
3. Converts the face position into the **angle of the target** in the room
4. **Predicts** where the target is now, compensating for camera and WiFi delay
5. **PID controller** turns the remaining error into pan/tilt turning speed
6. A **state machine** decides what to do: SCANNING → TRACKING → LOCKED

## Features
- **Robust detection:** YuNet handles head tilts and angles that a classic Haar cascade misses
- **Sticky targeting:** keeps following the same person, even if someone else steps in
- **Smoothing + deadband:** filters detection jitter so the turret holds still when the target does (no servo buzz)
- **Lock-on with hysteresis:** the lock zone scales with face size and gets bigger once locked
- **Zigzag scan mode:** when nobody is visible, sweeps row by row, starting in the direction the target was last heading
- **Latency compensation:** a Smith-predictor-style correction (accounts for how far the turret moved since the frame was taken) plus velocity lead (aims where the target is heading)
- **Virtual turret simulator:** a cropped "turret view" moves around the webcam image like a real camera, with simulated delay, so the control can be tuned before any hardware exists
- **Live PID tuning:** sliders adjust KP, KI, and KD while it runs; press `p` to toggle prediction and compare

## Results so far
| Metric | Value |
|---|---|
| Tracking speed (laptop) | ~40–50 FPS |
| Tuned gains (in simulation) | KP = 3.0, KI = 0, KD = 0.06 |
| Simulated latency | 200 ms |

### Engineering notes
- **Overshoot from delay:** with latency, a high KP alone oscillates. KD damps it.
- **Freezing while locked hurt tracking:** the turret lost fast-moving targets, so it now keeps correcting at all times and "locked" is only a status.
- **KI stays at 0:** the turret's angle already builds up over time, which removes steady-state error on its own.
- **Prediction:** at 350 ms of delay, tracking without prediction trails behind and overshoots, while prediction keeps the aim ahead of the target and settles faster.
- **Jitter at rest:** detector noise of a few pixels caused constant small corrections, so a deadband on both error and target speed fixed it.

## Files
| File | What it does |
|---|---|
| `face_track.py` | Face detection + tracking on a fixed webcam |
| `sim_turret.py` | Virtual turret with latency, scan mode, prediction, and live PID tuning |
| `face_detection_yunet_2023mar.onnx` | YuNet face detection model (from OpenCV Zoo) |

## Run it
```
pip install "opencv-python<5"
python sim_turret.py
```
Two windows open: **World** (the full scene, with the turret's aim shown as a box) and **Turret view** (what the turret sees, plus PID sliders). Press `p` to toggle prediction, and `q` to quit.

## Hardware (planned)
- Freenove ESP32-S3 camera board
- 2× MG90S metal-gear servos (pan + tilt)
- Active buzzer + LED for lock-on
- 3D-printed pan/tilt mount

## Roadmap
- [x] Face detection + tracking on webcam
- [x] PID control with smoothing and sticky targeting
- [x] Virtual turret simulator with latency + live tuning
- [x] Zigzag scan mode
- [x] Latency compensation (prediction) + deadband
- [ ] Stream ESP32-S3 camera to laptop over WiFi
- [ ] Drive real servos from the tracking output
- [ ] Re-tune PID on real hardware and measure real latency
- [ ] 3D-printed mount + lock-on LED/buzzer
- [ ] Demo video

## Built with
Python · OpenCV · YuNet · PID control · ESP32-S3 · Arduino C++
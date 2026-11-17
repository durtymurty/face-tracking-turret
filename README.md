# Face-Tracking Turret

A pan/tilt turret that finds a face with computer vision and keeps it centered using PID control with latency compensation.

**Status:** 🚧 In progress. The ESP32-S3 camera streams to the laptop over WiFi and face tracking runs live on it. Servo wiring is next.

## How it works
1. **ESP32-S3** captures video and streams it over WiFi
2. The laptop runs the **YuNet** neural-network face detector (OpenCV)
3. Converts the face position into the **angle of the target** in the room
4. **Predicts** where the target is now, compensating for camera and WiFi delay
5. **PID controller** turns the remaining error into pan/tilt turning speed
6. A **state machine** decides what to do: SCANNING → TRACKING → LOCKED

## Features
- **Robust detection:** YuNet handles head tilts and angles that a classic Haar cascade misses
- **Sticky targeting:** keeps following the same person, even if someone else steps in
- **Smoothing + deadband:** filters detection jitter so the turret holds still when the target does (no servo buzz)
- **Lock-on with hysteresis:** the lock zone scales with face size and gets bigger once locked
- **Zigzag scan mode:** sweeps row by row when nobody is visible, starting in the direction the target was last heading
- **Latency compensation:** a Smith-predictor-style correction (accounts for how far the turret moved since the frame was taken) plus velocity lead (aims where the target is heading)
- **Hardware-independent design:** `Tracker` (the brain) is separate from the turret, so the simulator and real hardware are swappable with one line
- **Virtual turret simulator:** tune the control with simulated delay before touching hardware, with live PID sliders and a `p` key to toggle prediction

## Results so far
| Metric | Value |
|---|---|
| Tracking speed (laptop webcam) | ~40–50 FPS |
| Tracking speed (ESP32-S3 camera over WiFi) | ~15–20 FPS |
| Tuned gains (in simulation) | KP = 3.0, KI = 0, KD = 0.06 |
| Simulated latency | 200 ms |

### Engineering notes
- **Overshoot from delay:** with latency, a high KP alone oscillates. KD damps it.
- **Freezing while locked hurt tracking:** the turret lost fast-moving targets, so it now keeps correcting at all times and "locked" is only a status.
- **KI stays at 0:** the turret's angle already builds up over time, which removes steady-state error on its own.
- **Prediction:** at 350 ms of delay, tracking without prediction trails behind and overshoots, while prediction keeps the aim ahead of the target and settles faster.
- **Jitter at rest:** detector noise of a few pixels caused constant small corrections, so a deadband on both error and target speed fixed it.
- **Camera sensor surprise:** the board shipped with a **GC0308** sensor (PID 0x9b), which has no hardware JPEG. Switched the camera to RGB565 and let the ESP32 encode JPEG in software.
- **Stream lag:** frames were queuing up, so the stream showed old frames. Fixed with `CAMERA_GRAB_LATEST` and a lower JPEG quality (80 → 50), which roughly doubled the FPS from ~9 to 15–20.

## Files
| File | What it does |
|---|---|
| `main.py` | Runs the loop: picks a turret (sim or real) and connects it to the tracker |
| `tracker.py` | The brain: detection, PID, scan mode, prediction (hardware-independent) |
| `sim_turret.py` | Virtual turret: webcam crop with simulated delay |
| `face_track.py` | Simple tracker; works with the laptop webcam or the ESP32 stream |
| `esp32_camera/` | ESP32-S3 firmware: camera + WiFi video stream (Arduino) |
| `face_detection_yunet_2023mar.onnx` | YuNet face detection model (from OpenCV Zoo) |

## Run it
**Simulator (laptop webcam only):**
```
pip install "opencv-python<5"
python main.py
```
Two windows open: **World** (the full scene, with the turret's aim shown as a box) and **Turret view** (what the turret sees, plus PID sliders). Press `p` to toggle prediction and `q` to quit.

**ESP32 camera:**
1. Copy `esp32_camera/secrets.example.h` to `secrets.h` and add your WiFi details (must be 2.4 GHz)
2. In the Arduino IDE, use board **ESP32S3 Dev Module**, PSRAM **OPI PSRAM**, Flash **8MB**, Partition **8M with spiffs**, USB CDC On Boot **Disabled**
3. Upload, open the Serial Monitor (115200), and copy the IP it prints
4. Put that IP in `face_track.py`, then run `python face_track.py`

## Hardware
- Freenove ESP32-S3 camera board (GC0308 sensor)
- 2× MG90S metal-gear servos (pan + tilt)
- Active buzzer + LED for lock-on
- 3D-printed pan/tilt mount (planned)

## Roadmap
- [x] Face detection + tracking on webcam
- [x] PID control with smoothing and sticky targeting
- [x] Virtual turret simulator with latency + live tuning
- [x] Zigzag scan mode
- [x] Latency compensation (prediction) + deadband
- [x] Refactor into hardware-independent tracker
- [x] Stream ESP32-S3 camera to laptop over WiFi
- [ ] Drive real servos from the tracking output
- [ ] Re-tune PID on real hardware and measure real latency
- [ ] 3D-printed mount + lock-on LED/buzzer
- [ ] Demo video

## Built with
Python · OpenCV · YuNet · PID control · ESP32-S3 · Arduino C++
# Face-Tracking Turret

A pan/tilt turret that finds a face with computer vision and keeps it centered using closed-loop control.

**Status:** 🚧 In progress — face detection + control logic working on laptop webcam; hardware build next.

## How it works
1. Camera captures video
2. YuNet neural-network face detector (OpenCV) finds the face
3. Calculates how far the face is from the center of the frame (error)
4. Proportional control converts that error into pan/tilt servo angles
5. Turret moves until the face is centered → "LOCKED"

## Hardware (planned)
- Freenove ESP32-S3 camera board
- 2× MG90S metal-gear servos (pan + tilt)
- Active buzzer + LED for lock-on
- 3D-printed pan/tilt mount

## Run it (laptop webcam demo)
```
pip install "opencv-python<5"
python face_track.py
```
Press `q` to quit.

## Roadmap
- [x] Face detection + tracking logic on webcam
- [ ] Stream ESP32-S3 camera to laptop over WiFi
- [ ] Drive servos from tracking output
- [ ] Tune control (PID) for smooth tracking
- [ ] 3D-printed mount + lock-on LED/buzzer
- [ ] Demo video

## Built with
Python · OpenCV · ESP32-S3 · Arduino C++

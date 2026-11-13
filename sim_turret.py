import cv2
import time
from collections import deque

# ---------- Setup ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Couldn't open webcam")

ok, frame = cap.read()
H, W = frame.shape[:2]          # full webcam image = "the world"
CW, CH = W // 2, H // 2         # turret camera only sees half the world
MAX_SPEED = 300                 # servo speed limit in degrees/second (real servos aren't instant)
LATENCY = 0.15                  # seconds of delay (WiFi + detection + servo movement on the real turret)

# Detector runs on the turret's view, so give it the view size
detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (CW, CH), 0.6)


class PID:
    """Error (pixels) in -> turning speed (degrees/second) out."""
    def __init__(self):
        self.kp = self.ki = self.kd = 0.0
        self.integral, self.prev_err = 0.0, 0.0

    def update(self, err, dt):
        self.integral = max(-200, min(200, self.integral + err * dt))   # anti-windup cap
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err
        return self.kp * err + self.ki * self.integral + self.kd * deriv

    def reset(self):
        self.integral, self.prev_err = 0.0, 0.0


def window_position(pan, tilt):
    """Convert servo angles (0-180) into where the turret's view box sits in the world."""
    x0 = int(pan / 180 * (W - CW))
    y0 = int(tilt / 180 * (H - CH))
    return x0, y0


# ---------- Sliders for live tuning ----------
VIEW = "Turret view"
cv2.namedWindow(VIEW)
cv2.createTrackbar("KP x100", VIEW, 300, 500, lambda v: None)   # tuned: KP = 3.0
cv2.createTrackbar("KI x100", VIEW, 0, 100, lambda v: None)     # leave at 0
cv2.createTrackbar("KD x1000", VIEW, 60, 200, lambda v: None)   # tuned: KD = 0.060

pid_pan, pid_tilt = PID(), PID()
pan, tilt = 90.0, 90.0          # start aimed at the center
history = deque()               # past angle commands, used to fake the delay
LOCK_RATIO, UNLOCK_MULT, SMOOTH = 0.4, 1.5, 0.7
track, locked, lost_frames = None, False, 0
prev_time, fps = time.time(), 0.0

# ---------- Main loop ----------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    now = time.time()
    dt = max(now - prev_time, 1e-3)
    prev_time = now
    fps = 0.9 * fps + 0.1 / dt

    # Read slider values into both PID controllers
    for pid in (pid_pan, pid_tilt):
        pid.kp = cv2.getTrackbarPos("KP x100", VIEW) / 100
        pid.ki = cv2.getTrackbarPos("KI x100", VIEW) / 100
        pid.kd = cv2.getTrackbarPos("KD x1000", VIEW) / 1000

    # 1) What does the turret see right now?
    # The view shows where the turret was aimed LATENCY seconds ago (like real hardware)
    history.append((now, pan, tilt))
    while len(history) > 1 and history[1][0] <= now - LATENCY:
        history.popleft()
    _, view_pan, view_tilt = history[0]
    x0, y0 = window_position(view_pan, view_tilt)
    view = frame[y0:y0 + CH, x0:x0 + CW].copy()
    cx, cy = CW // 2, CH // 2

    # 2) Find the face in the turret's view
    _, faces = detector.detect(view)
    if faces is not None and len(faces):
        boxes = [f[:4] for f in faces]
        if track is None:
            target = max(boxes, key=lambda b: b[2] * b[3])
        else:
            tx, ty = track[0] + track[2] / 2, track[1] + track[3] / 2
            target = min(boxes, key=lambda b: (b[0] + b[2] / 2 - tx) ** 2 + (b[1] + b[3] / 2 - ty) ** 2)
        track = target if track is None else SMOOTH * target + (1 - SMOOTH) * track
        lost_frames = 0
    else:
        lost_frames += 1
        if lost_frames > 10:
            track, locked = None, False
            pid_pan.reset(); pid_tilt.reset()

    # 3) Turn toward it
    status, color = "SEARCHING", (0, 0, 255)
    if track is not None:
        x, y, fw, fh = track.astype(int)
        fx, fy = x + fw // 2, y + fh // 2
        err_x, err_y = fx - cx, fy - cy

        zone = LOCK_RATIO * fw * (UNLOCK_MULT if locked else 1.0)
        locked = abs(err_x) < zone and abs(err_y) < zone

        if locked:
            status, color = "LOCKED", (0, 255, 0)
        else:
            status, color = "TRACKING", (0, 255, 255)

        # Always keep correcting; "locked" is just a status now, not a freeze
        pan_speed = max(-MAX_SPEED, min(MAX_SPEED, pid_pan.update(err_x, dt)))
        tilt_speed = max(-MAX_SPEED, min(MAX_SPEED, pid_tilt.update(err_y, dt)))
        pan = max(0, min(180, pan + pan_speed * dt))      # speed x time = how far it turned
        tilt = max(0, min(180, tilt + tilt_speed * dt))

        cv2.rectangle(view, (x, y), (x + fw, y + fh), color, 2)
        cv2.line(view, (cx, cy), (fx, fy), color, 1)

    # 4) Draw everything
    cv2.drawMarker(view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    cv2.putText(view, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.rectangle(frame, (x0, y0), (x0 + CW, y0 + CH), color, 2)   # where the turret is aimed
    cv2.putText(frame, f"pan={pan:.0f}  tilt={tilt:.0f}   FPS {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("World", frame)
    cv2.imshow(VIEW, cv2.resize(view, (CW * 2, CH * 2)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
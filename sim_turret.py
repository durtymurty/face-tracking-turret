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
LATENCY = 0.2                  # seconds of delay (WiFi + detection + servo movement on the real turret)
PX_PER_DEG_X = (W - CW) / 180   # how many pixels the view shifts per degree of pan
PX_PER_DEG_Y = (H - CH) / 180   # same for tilt (on the real turret this comes from the camera's field of view)

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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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
SCAN_DELAY = 1.5                # seconds without a face before scanning starts
SCAN_SPEED = 60                 # degrees/second while sweeping
SCAN_MIN, SCAN_MAX = 10, 170    # sweep limits (left/right)
SCAN_ROWS = [90, 0, 180]        # tilt angles to sweep: middle, top, bottom
last_seen, scan_dir = time.time(), 1   # when we last saw a face, sweep direction (1 = right, -1 = left)
scan_row, scanning, last_dir = 0, False, 1
track, locked, lost_frames = None, False, 0
prev_time, fps = time.time(), 0.0

PREDICT = True                  # press P to toggle and compare
DEADBAND_PX = 8                 # ignore errors smaller than this (detector noise)
VEL_DEADBAND = 5                # ignore target speeds below this (deg/s) so sitting still = still
prev_target = None              # target's angle last frame
vel_pan = vel_tilt = 0.0        # target's estimated speed (degrees/second)

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
        last_seen = now
    else:
        lost_frames += 1
        if lost_frames > 10:
            track, locked = None, False
            prev_target, vel_pan, vel_tilt = None, 0.0, 0.0
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

        # --- Prediction ---
        # Convert "pixels off-center in an old frame" into "what angle is the target at in the room".
        # The frame was taken when the turret pointed at view_pan/view_tilt (LATENCY seconds ago).
        target_pan = view_pan + err_x / PX_PER_DEG_X
        target_tilt = view_tilt + err_y / PX_PER_DEG_Y

        # Estimate how fast the target is moving (degrees/second), smoothed to ignore jitter
        if prev_target is not None:
            vel_pan = clamp(0.7 * vel_pan + 0.3 * (target_pan - prev_target[0]) / dt, -200, 200)
            vel_tilt = clamp(0.7 * vel_tilt + 0.3 * (target_tilt - prev_target[1]) / dt, -200, 200)
        prev_target = (target_pan, target_tilt)

        if PREDICT:
            # Lead the target by the delay, then compare to where the turret points NOW (Smith predictor idea)
            lead_pan = vel_pan if abs(vel_pan) > VEL_DEADBAND else 0.0
            lead_tilt = vel_tilt if abs(vel_tilt) > VEL_DEADBAND else 0.0
            aim_pan = target_pan + lead_pan * LATENCY
            aim_tilt = target_tilt + lead_tilt * LATENCY
            ctrl_x = (aim_pan - pan) * PX_PER_DEG_X
            ctrl_y = (aim_tilt - tilt) * PX_PER_DEG_Y
        else:
            ctrl_x, ctrl_y = err_x, err_y                 # old way: react to the stale frame
        # Deadband: tiny errors are just noise, so don't twitch the servos
        if abs(ctrl_x) < DEADBAND_PX:
            ctrl_x = 0
        if abs(ctrl_y) < DEADBAND_PX:
            ctrl_y = 0
        # Always keep correcting; "locked" is just a status, not a freeze
        pan_speed = clamp(pid_pan.update(ctrl_x, dt), -MAX_SPEED, MAX_SPEED)
        tilt_speed = clamp(pid_tilt.update(ctrl_y, dt), -MAX_SPEED, MAX_SPEED)
        pan = clamp(pan + pan_speed * dt, 0, 180)          # speed x time = how far it turned
        tilt = clamp(tilt + tilt_speed * dt, 0, 180)
        last_dir = 1 if pan_speed >= 0 else -1             # which way they were heading
        scanning = False

        cv2.rectangle(view, (x, y), (x + fw, y + fh), color, 2)
        cv2.line(view, (cx, cy), (fx, fy), color, 1)

    # 3b) Nobody found for a while -> zigzag scan
    if track is None and now - last_seen > SCAN_DELAY:
        if not scanning:                                  # just started scanning
            scanning, scan_dir, scan_row = True, last_dir, 0   # head where they were going first
        status, color = "SCANNING", (255, 128, 0)
        pan += scan_dir * SCAN_SPEED * dt
        if pan >= SCAN_MAX or pan <= SCAN_MIN:            # hit the edge -> turn around, next row
            pan = clamp(pan, SCAN_MIN, SCAN_MAX)
            scan_dir *= -1
            scan_row = (scan_row + 1) % len(SCAN_ROWS)
        target_row = SCAN_ROWS[scan_row]
        tilt += (target_row - tilt) * min(1, 3 * dt)      # glide to this row's height
    # Show where the turret is aiming (dot) vs where the target actually is (circle)
    if track is not None:
        aim_x = int(aim_pan * PX_PER_DEG_X + CW / 2) if PREDICT else int(target_pan * PX_PER_DEG_X + CW / 2)
        aim_y = int(aim_tilt * PX_PER_DEG_Y + CH / 2) if PREDICT else int(target_tilt * PX_PER_DEG_Y + CH / 2)
        real_x = int(target_pan * PX_PER_DEG_X + CW / 2)
        real_y = int(target_tilt * PX_PER_DEG_Y + CH / 2)
        cv2.circle(frame, (real_x, real_y), 12, (255, 255, 255), 2)   # where the old frame says you are
        cv2.circle(frame, (aim_x, aim_y), 6, (0, 255, 0), -1)         # where the turret is aiming
    # 4) Draw everything
    cv2.drawMarker(view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    cv2.putText(view, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.rectangle(frame, (x0, y0), (x0 + CW, y0 + CH), color, 2)   # where the turret is aimed
    cv2.putText(frame, f"pan={pan:.0f}  tilt={tilt:.0f}   FPS {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"PREDICTION {'ON' if PREDICT else 'OFF'} (press p)", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if PREDICT else (0, 0, 255), 2)

    cv2.imshow("World", frame)
    cv2.imshow(VIEW, cv2.resize(view, (CW * 2, CH * 2)))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("p"):
        PREDICT = not PREDICT

cap.release()
cv2.destroyAllWindows()
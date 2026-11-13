import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Couldn't open webcam")

ok, frame = cap.read()
h, w = frame.shape[:2]
cx, cy = w // 2, h // 2

detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (w, h), 0.6)


class PID:
    """Turns pixel error into a servo angle change."""
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral, self.prev_err = 0.0, 0.0

    def update(self, err, dt):
        self.integral = max(-500, min(500, self.integral + err * dt))   # cap so it can't wind up
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err
        return self.kp * err + self.ki * self.integral + self.kd * deriv

    def reset(self):
        self.integral, self.prev_err = 0.0, 0.0


pid_pan = PID(kp=0.02, ki=0.001, kd=0.004)
pid_tilt = PID(kp=0.02, ki=0.001, kd=0.004)

pan, tilt = 90.0, 90.0
LOCK_RATIO = 0.4          # locked if face center is within 40% of face width from center
UNLOCK_MULT = 1.5         # must drift 1.5x farther to lose lock (stops flickering)
SMOOTH = 0.4              # 0-1: lower = smoother but slower to react
track = None              # smoothed [x, y, w, h] of the face we're following
locked = False
lost_frames = 0
prev_time = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = 0.9 * fps + 0.1 * (1 / dt if dt > 0 else 0)

    _, faces = detector.detect(frame)
    if faces is not None and len(faces):
        boxes = [f[:4] for f in faces]
        if track is None:
            target = max(boxes, key=lambda b: b[2] * b[3])           # first time: biggest face
        else:
            tx, ty = track[0] + track[2] / 2, track[1] + track[3] / 2
            target = min(boxes, key=lambda b: (b[0] + b[2] / 2 - tx) ** 2 + (b[1] + b[3] / 2 - ty) ** 2)  # stick to same person
        track = target if track is None else SMOOTH * target + (1 - SMOOTH) * track
        lost_frames = 0
    else:
        lost_frames += 1
        if lost_frames > 10:
            track, locked = None, False
            pid_pan.reset(); pid_tilt.reset()

    status, color = "SEARCHING", (0, 0, 255)
    if track is not None:
        x, y, fw, fh = track.astype(int)
        fx, fy = x + fw // 2, y + fh // 2
        err_x, err_y = fx - cx, fy - cy

        zone = LOCK_RATIO * fw * (UNLOCK_MULT if locked else 1.0)   # bigger zone once locked
        locked = abs(err_x) < zone and abs(err_y) < zone

        if locked:
            status, color = "LOCKED", (0, 255, 0)
        else:
            status, color = "TRACKING", (0, 255, 255)
            pan = max(0, min(180, pan + pid_pan.update(err_x, dt)))
            tilt = max(0, min(180, tilt + pid_tilt.update(err_y, dt)))

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        cv2.line(frame, (cx, cy), (fx, fy), color, 2)
        cv2.circle(frame, (cx, cy), int(zone), color, 1)          # shows the lock zone
        cv2.putText(frame, f"error x={err_x} y={err_y}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"pan={pan:.0f}  tilt={tilt:.0f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Turret - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
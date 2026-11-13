import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Couldn't open webcam")

ok, frame = cap.read()
h, w = frame.shape[:2]

# YuNet: small neural-network face detector (handles tilted/turned faces)
detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (w, h), 0.6)

pan, tilt = 90.0, 90.0   # simulated servo angles
KP = 0.02                # correction strength
DEADZONE = 30            # pixels = "locked"
last_face, lost_frames = None, 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    cx, cy = w // 2, h // 2

    _, faces = detector.detect(frame)
    if faces is not None and len(faces):
        best = max(faces, key=lambda f: f[2] * f[3])        # biggest face
        last_face, lost_frames = best[:4].astype(int), 0
    else:
        lost_frames += 1
        if lost_frames > 10:                                # forget after ~10 missed frames
            last_face = None

    status, color = "SEARCHING", (0, 0, 255)
    if last_face is not None:
        x, y, fw, fh = last_face
        fx, fy = x + fw // 2, y + fh // 2
        err_x, err_y = fx - cx, fy - cy

        if abs(err_x) < DEADZONE and abs(err_y) < DEADZONE:
            status, color = "LOCKED", (0, 255, 0)
        else:
            status, color = "TRACKING", (0, 255, 255)
            pan = max(0, min(180, pan + KP * err_x))
            tilt = max(0, min(180, tilt + KP * err_y))

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        cv2.line(frame, (cx, cy), (fx, fy), color, 2)
        cv2.putText(frame, f"error x={err_x} y={err_y}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"pan={pan:.0f}  tilt={tilt:.0f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Turret - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
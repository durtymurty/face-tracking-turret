import time
import cv2
from tracker import Tracker
from sim_turret import SimTurret
# from real_turret import RealTurret      # coming when the parts arrive

# ===== Pick your turret: this is the ONLY line that changes for real hardware =====
turret = SimTurret(latency=0.2)
# turret = RealTurret(...)

tracker = Tracker("face_detection_yunet_2023mar.onnx", turret.view_size, turret.px_per_deg, turret.latency)

COLORS = {"SEARCHING": (0, 0, 255), "SCANNING": (255, 128, 0),
          "TRACKING": (0, 255, 255), "LOCKED": (0, 255, 0)}

VIEW = "Turret view"
cv2.namedWindow(VIEW)
cv2.createTrackbar("KP x100", VIEW, 300, 500, lambda v: None)
cv2.createTrackbar("KI x100", VIEW, 0, 100, lambda v: None)
cv2.createTrackbar("KD x1000", VIEW, 60, 200, lambda v: None)

prev_time, fps = time.time(), 0.0

while True:
    now = time.time()
    dt = max(now - prev_time, 1e-3)
    prev_time = now
    fps = 0.9 * fps + 0.1 / dt

    tracker.set_gains(cv2.getTrackbarPos("KP x100", VIEW) / 100,
                      cv2.getTrackbarPos("KI x100", VIEW) / 100,
                      cv2.getTrackbarPos("KD x1000", VIEW) / 1000)

    # The whole loop: see -> think -> move
    view, view_pan, view_tilt = turret.get_frame()
    if view is None:
        break
    pan, tilt, info = tracker.update(view, view_pan, view_tilt, dt, now)
    turret.move(pan, tilt)

    # Draw the turret's view
    color = COLORS[tracker.status]
    cx, cy = tracker.cx, tracker.cy
    if info["box"]:
        x, y, fw, fh = info["box"]
        cv2.rectangle(view, (x, y), (x + fw, y + fh), color, 2)
        cv2.line(view, (cx, cy), (x + fw // 2, y + fh // 2), color, 1)
    cv2.drawMarker(view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    cv2.putText(view, tracker.status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow(VIEW, cv2.resize(view, (view.shape[1] * 2, view.shape[0] * 2)))

    pred_color = (0, 255, 0) if tracker.predict else (0, 0, 255)      # green ON, red OFF
    turret.show(color, info, [(f"pan={pan:.0f}  tilt={tilt:.0f}   FPS {fps:.0f}", (255, 255, 255)),
                              (f"PREDICTION {'ON' if tracker.predict else 'OFF'} (press p)", pred_color)])

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("p"):
        tracker.predict = not tracker.predict

turret.close()
cv2.destroyAllWindows()
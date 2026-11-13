import time
import cv2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PID:
    """Error (pixels) in -> turning speed (degrees/second) out."""
    def __init__(self):
        self.kp = self.ki = self.kd = 0.0
        self.integral, self.prev_err = 0.0, 0.0

    def update(self, err, dt):
        self.integral = clamp(self.integral + err * dt, -200, 200)   # anti-windup cap
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err
        return self.kp * err + self.ki * self.integral + self.kd * deriv

    def reset(self):
        self.integral, self.prev_err = 0.0, 0.0


class Tracker:
    """The brain: finds a face and decides where the turret should point.
    Knows nothing about hardware -- works the same with the simulator or the real turret."""

    def __init__(self, model_path, view_size, px_per_deg, latency):
        w, h = view_size
        self.cx, self.cy = w // 2, h // 2
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.6)
        self.px_x, self.px_y = px_per_deg      # pixels the view shifts per degree of pan/tilt
        self.latency = latency                 # seconds between capturing a frame and acting on it
        self.pid_pan, self.pid_tilt = PID(), PID()

        # --- Settings ---
        self.max_speed = 300                   # deg/s servo limit
        self.lock_ratio, self.unlock_mult, self.smooth = 0.4, 1.5, 0.7
        self.scan_delay, self.scan_speed = 1.5, 60
        self.scan_min, self.scan_max = 10, 170
        self.scan_rows = [90, 0, 180]
        self.deadband_px, self.vel_deadband = 8, 5
        self.predict = True

        # --- State ---
        self.pan, self.tilt = 90.0, 90.0       # where we're commanding the turret to point
        self.track, self.locked, self.lost_frames = None, False, 0
        self.last_seen = time.time()
        self.scan_dir, self.scan_row, self.scanning, self.last_dir = 1, 0, False, 1
        self.prev_target, self.vel_pan, self.vel_tilt = None, 0.0, 0.0
        self.status = "SEARCHING"

    def set_gains(self, kp, ki, kd):
        for pid in (self.pid_pan, self.pid_tilt):
            pid.kp, pid.ki, pid.kd = kp, ki, kd

    def update(self, view, view_pan, view_tilt, dt, now):
        """One step: given a camera frame (and the angles it was taken at), return new (pan, tilt, info)."""
        info = {"box": None, "target": None, "aim": None}

        # 1) Find the face
        _, faces = self.detector.detect(view)
        if faces is not None and len(faces):
            boxes = [f[:4] for f in faces]
            if self.track is None:
                target = max(boxes, key=lambda b: b[2] * b[3])          # first time: biggest face
            else:
                tx = self.track[0] + self.track[2] / 2
                ty = self.track[1] + self.track[3] / 2
                target = min(boxes, key=lambda b: (b[0] + b[2] / 2 - tx) ** 2 + (b[1] + b[3] / 2 - ty) ** 2)
            self.track = target if self.track is None else self.smooth * target + (1 - self.smooth) * self.track
            self.lost_frames = 0
            self.last_seen = now
        else:
            self.lost_frames += 1
            if self.lost_frames > 10:
                self.track, self.locked = None, False
                self.prev_target, self.vel_pan, self.vel_tilt = None, 0.0, 0.0
                self.pid_pan.reset(); self.pid_tilt.reset()

        # 2) State machine: follow, scan, or wait
        self.status = "SEARCHING"
        if self.track is not None:
            self._follow(view_pan, view_tilt, dt, info)
        elif now - self.last_seen > self.scan_delay:
            self._scan(dt)

        return self.pan, self.tilt, info

    def _follow(self, view_pan, view_tilt, dt, info):
        x, y, fw, fh = self.track.astype(int)
        err_x, err_y = x + fw // 2 - self.cx, y + fh // 2 - self.cy

        zone = self.lock_ratio * fw * (self.unlock_mult if self.locked else 1.0)
        self.locked = abs(err_x) < zone and abs(err_y) < zone
        self.status = "LOCKED" if self.locked else "TRACKING"

        # Where is the target in the room (angles)? The frame was taken at view_pan/view_tilt.
        target_pan = view_pan + err_x / self.px_x
        target_tilt = view_tilt + err_y / self.px_y
        if self.prev_target is not None:
            self.vel_pan = clamp(0.7 * self.vel_pan + 0.3 * (target_pan - self.prev_target[0]) / dt, -200, 200)
            self.vel_tilt = clamp(0.7 * self.vel_tilt + 0.3 * (target_tilt - self.prev_target[1]) / dt, -200, 200)
        self.prev_target = (target_pan, target_tilt)

        if self.predict:
            lead_pan = self.vel_pan if abs(self.vel_pan) > self.vel_deadband else 0.0
            lead_tilt = self.vel_tilt if abs(self.vel_tilt) > self.vel_deadband else 0.0
            aim_pan = target_pan + lead_pan * self.latency
            aim_tilt = target_tilt + lead_tilt * self.latency
            ctrl_x = (aim_pan - self.pan) * self.px_x
            ctrl_y = (aim_tilt - self.tilt) * self.px_y
        else:
            aim_pan, aim_tilt = target_pan, target_tilt
            ctrl_x, ctrl_y = err_x, err_y

        if abs(ctrl_x) < self.deadband_px:
            ctrl_x = 0
        if abs(ctrl_y) < self.deadband_px:
            ctrl_y = 0

        pan_speed = clamp(self.pid_pan.update(ctrl_x, dt), -self.max_speed, self.max_speed)
        tilt_speed = clamp(self.pid_tilt.update(ctrl_y, dt), -self.max_speed, self.max_speed)
        self.pan = clamp(self.pan + pan_speed * dt, 0, 180)
        self.tilt = clamp(self.tilt + tilt_speed * dt, 0, 180)
        self.last_dir = 1 if pan_speed >= 0 else -1
        self.scanning = False

        info["box"] = (x, y, fw, fh)
        info["target"] = (target_pan, target_tilt)
        info["aim"] = (aim_pan, aim_tilt)

    def _scan(self, dt):
        if not self.scanning:                                  # just started: head where they went
            self.scanning, self.scan_dir, self.scan_row = True, self.last_dir, 0
        self.status = "SCANNING"
        self.pan += self.scan_dir * self.scan_speed * dt
        if self.pan >= self.scan_max or self.pan <= self.scan_min:
            self.pan = clamp(self.pan, self.scan_min, self.scan_max)
            self.scan_dir *= -1
            self.scan_row = (self.scan_row + 1) % len(self.scan_rows)
        target_row = self.scan_rows[self.scan_row]
        self.tilt += (target_row - self.tilt) * min(1, 3 * dt)
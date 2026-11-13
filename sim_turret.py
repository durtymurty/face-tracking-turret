import time
from collections import deque
import cv2


class SimTurret:
    """Fake turret: crops a window out of the webcam image and moves it like a camera on servos."""

    def __init__(self, latency=0.2, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise SystemExit("Couldn't open webcam")
        ok, frame = self.cap.read()
        self.H, self.W = frame.shape[:2]
        self.view_size = (self.W // 2, self.H // 2)            # turret sees half the world
        cw, ch = self.view_size
        self.px_per_deg = ((self.W - cw) / 180, (self.H - ch) / 180)
        self.latency = latency
        self.pan, self.tilt = 90.0, 90.0
        self.history = deque()                                 # past commands, used to fake delay
        self.world, self.x0, self.y0 = None, 0, 0

    def _window(self, pan, tilt):
        cw, ch = self.view_size
        return int(pan / 180 * (self.W - cw)), int(tilt / 180 * (self.H - ch))

    def get_frame(self):
        """Returns (view image, pan, tilt) -- the angles the 'camera' was at when the frame was taken."""
        ok, frame = self.cap.read()
        if not ok:
            return None, 0, 0
        frame = cv2.flip(frame, 1)
        now = time.time()
        self.history.append((now, self.pan, self.tilt))
        while len(self.history) > 1 and self.history[1][0] <= now - self.latency:
            self.history.popleft()
        _, view_pan, view_tilt = self.history[0]
        self.x0, self.y0 = self._window(view_pan, view_tilt)
        cw, ch = self.view_size
        self.world = frame
        return frame[self.y0:self.y0 + ch, self.x0:self.x0 + cw].copy(), view_pan, view_tilt

    def move(self, pan, tilt):
        """Fake servos: just remember the command (the view catches up after the delay)."""
        self.pan, self.tilt = pan, tilt

    def show(self, color, info, lines):
        """Draw the 'World' window: full scene, where the turret is aimed, target vs aim point."""
        f = self.world
        cw, ch = self.view_size
        px, py = self.px_per_deg
        cv2.rectangle(f, (self.x0, self.y0), (self.x0 + cw, self.y0 + ch), color, 2)
        if info["target"]:
            tx, ty = info["target"]
            ax, ay = info["aim"]
            cv2.circle(f, (int(tx * px + cw / 2), int(ty * py + ch / 2)), 12, (255, 255, 255), 2)
            cv2.circle(f, (int(ax * px + cw / 2), int(ay * py + ch / 2)), 6, (0, 255, 0), -1)
        for i, (text, text_color) in enumerate(lines):      # each line is (text, color)
            cv2.putText(f, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.imshow("World", f)

    def close(self):
        self.cap.release()
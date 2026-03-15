import threading
import time

import cv2
import numpy as np

from core.utils import get_colormap

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - depends on device runtime
    rs = None
    REALSENSE_IMPORT_ERROR = exc
else:
    REALSENSE_IMPORT_ERROR = None


class G1CameraManager:
    VALID_STREAMS = {"color", "depth", "combined"}

    def __init__(self, cfg):
        self.cfg = cfg
        self.source = str(cfg.get("source", "realsense")).lower()
        self.webcam_index = int(cfg.get("webcam_index", 0))
        self.width = int(cfg.get("width", 640))
        self.height = int(cfg.get("height", 480))
        self.fps = int(cfg.get("fps", 30))
        self.enable_rgb = bool(cfg.get("enable_rgb", True))
        self.enable_depth = bool(cfg.get("enable_depth", True))
        self.align_depth = bool(cfg.get("align_depth", True))
        self.jpeg_quality = int(cfg.get("jpeg_quality", 85))
        self.depth_colormap = get_colormap(cfg.get("depth_colormap", "JET"))
        self.depth_mode = "real" if self.source == "realsense" else "mock"

        self.pipeline = rs.pipeline() if rs is not None else None
        self.rs_config = rs.config() if rs is not None else None
        self.align = None
        self.capture = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._started = False
        self._frame_counter = 0
        self._last_error = None
        self._frames = {
            "color": None,
            "depth": None,
            "combined": None,
        }

        if not self.enable_rgb and not self.enable_depth:
            self._last_error = "camera.enable_rgb 和 camera.enable_depth 不能同时为 false。"
            return

        if self.source not in {"realsense", "webcam"}:
            self._last_error = f"不支持的 camera.source: {self.source}"
            return

        if self.source == "realsense":
            if rs is None:
                self._last_error = f"pyrealsense2 未安装: {REALSENSE_IMPORT_ERROR}"
                return

            if self.enable_rgb:
                self.rs_config.enable_stream(
                    rs.stream.color,
                    self.width,
                    self.height,
                    rs.format.bgr8,
                    self.fps,
                )

            if self.enable_depth:
                self.rs_config.enable_stream(
                    rs.stream.depth,
                    self.width,
                    self.height,
                    rs.format.z16,
                    self.fps,
                )

            if self.enable_rgb and self.enable_depth and self.align_depth:
                self.align = rs.align(rs.stream.color)

    def start(self):
        if self._started:
            return self

        if self.source == "realsense":
            self._start_realsense()
        else:
            self._start_webcam()

        return self

    def _start_realsense(self):
        if self.pipeline is None or self.rs_config is None:
            return

        try:
            self.pipeline.start(self.rs_config)
            self._started = True
            self._thread = threading.Thread(target=self._update_realsense_loop, daemon=True)
            self._thread.start()
        except Exception as exc:  # pragma: no cover - depends on device runtime
            self._last_error = f"启动 RealSense 相机失败: {exc}"

    def _start_webcam(self):
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else 0
        self.capture = cv2.VideoCapture(self.webcam_index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(self.webcam_index)

        if not self.capture.isOpened():
            self._last_error = f"无法打开本机摄像头 index={self.webcam_index}"
            return

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._started = True
        self._thread = threading.Thread(target=self._update_webcam_loop, daemon=True)
        self._thread.start()

    def _update_realsense_loop(self):
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if self.align is not None:
                    frames = self.align.process(frames)

                color_frame = frames.get_color_frame() if self.enable_rgb else None
                depth_frame = frames.get_depth_frame() if self.enable_depth else None

                color_image = (
                    np.asanyarray(color_frame.get_data()) if color_frame else None
                )
                depth_image = (
                    np.asanyarray(depth_frame.get_data()) if depth_frame else None
                )
                depth_visual = (
                    self._build_depth_visual(depth_image) if depth_image is not None else None
                )
                combined_image = self._build_combined(color_image, depth_visual)

                with self._lock:
                    self._frames["color"] = color_image
                    self._frames["depth"] = depth_visual
                    self._frames["combined"] = combined_image
                    self._frame_counter += 1
                    self._last_error = None
            except Exception as exc:  # pragma: no cover - depends on device runtime
                self._last_error = f"读取相机帧失败: {exc}"
                time.sleep(0.2)

    def _update_webcam_loop(self):
        while not self._stop_event.is_set():
            try:
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    raise RuntimeError("OpenCV 未读到摄像头画面")

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                color_image = frame if self.enable_rgb else None
                depth_visual = (
                    self._build_mock_depth_visual(frame) if self.enable_depth else None
                )
                combined_image = self._build_combined(color_image, depth_visual)

                with self._lock:
                    self._frames["color"] = color_image
                    self._frames["depth"] = depth_visual
                    self._frames["combined"] = combined_image
                    self._frame_counter += 1
                    self._last_error = None
            except Exception as exc:
                self._last_error = f"读取本机摄像头失败: {exc}"
                time.sleep(0.2)

    def _build_depth_visual(self, depth_image):
        normalized = cv2.normalize(
            depth_image,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        normalized[depth_image == 0] = 0
        return cv2.applyColorMap(normalized, self.depth_colormap)

    def _build_mock_depth_visual(self, color_image):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        inverted = cv2.bitwise_not(normalized)
        return cv2.applyColorMap(inverted, self.depth_colormap)

    def _build_combined(self, color_image, depth_visual):
        if color_image is None:
            return depth_visual
        if depth_visual is None:
            return color_image

        if color_image.shape[:2] != depth_visual.shape[:2]:
            depth_visual = cv2.resize(
                depth_visual,
                (color_image.shape[1], color_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return np.hstack((color_image, depth_visual))

    def _build_placeholder(self, stream_name):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        title = {
            "depth": "Depth Stream Waiting",
            "color": "RGB Stream Waiting",
            "combined": "Combined Stream Waiting",
        }.get(stream_name, "Camera Stream Waiting")

        cv2.putText(
            canvas,
            title,
            (24, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        detail = self._last_error or "正在等待 G1 第一视角相机输出..."
        cv2.putText(
            canvas,
            detail[:60],
            (24, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"source={self.source} depth_mode={self.depth_mode}",
            (24, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 220, 120),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"{self.width}x{self.height} @ {self.fps} FPS",
            (24, self.height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def get_encoded_frame(self, stream_name="depth"):
        if stream_name not in self.VALID_STREAMS:
            stream_name = "depth"

        with self._lock:
            frame = self._frames.get(stream_name)
            last_error = self._last_error

        if frame is None:
            frame = self._build_placeholder(stream_name)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            self._last_error = last_error or "JPEG 编码失败。"
            return None
        return buffer.tobytes()

    def get_status(self):
        with self._lock:
            available_streams = [
                stream_name
                for stream_name, frame in self._frames.items()
                if frame is not None
            ]

            return {
                "running": self._started and not self._stop_event.is_set(),
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "camera_source": self.source,
                "depth_mode": self.depth_mode,
                "webcam_index": self.webcam_index,
                "enable_rgb": self.enable_rgb,
                "enable_depth": self.enable_depth,
                "align_depth": self.align_depth,
                "available_streams": available_streams,
                "frame_counter": self._frame_counter,
                "last_error": self._last_error,
            }

    def stop(self):
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        if self.pipeline is not None and self._started and self.source == "realsense":
            try:
                self.pipeline.stop()
            except Exception:  # pragma: no cover - depends on device runtime
                pass
        if self.capture is not None:
            self.capture.release()
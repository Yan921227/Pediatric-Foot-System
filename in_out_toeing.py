# foot_pose_all_in_one.py
# 整合：overlay.py + analyzer.py + pose_backend.py + gui_app.py
# 需求：PyQt5, opencv-python, mediapipe, numpy

from __future__ import annotations

import os
from collections import Counter, deque
from typing import Any, Dict, Mapping, Sequence, Tuple, Optional

import cv2
import numpy as np
from math import acos, degrees
from PyQt5 import QtCore, QtGui, QtWidgets


# ───────────────────────────────────────────────────────────────
# overlay.py 內容（抽成函式）
def draw_arrow(image, origin, vec, color=(0, 255, 0), scale=100, thickness=2):
    pt1 = tuple(np.round(origin[:2]).astype(int))
    pt2 = tuple(np.round((origin[:2] + vec[:2] * scale)).astype(int))
    cv2.arrowedLine(image, pt1, pt2, color, thickness, tipLength=0.3)


def draw_text_info(image, pos, label_dict, side: str):
    x, y = pos
    lines = [
        f"{side.title()} Leg: {label_dict['status']}",
        f"Angle: {label_dict['foot_angle_deg']:.2f}",
        f"HipStatus: {label_dict['hip_rotation_status']}",
        f"HipRot: {label_dict['hip_rotation_deg']:.2f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            image, line, (x, y + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3
        )


def draw_body_forward(image, center, forward_vec, scale=100, color=(255, 0, 255)):
    origin = np.array([center[0], center[1]])
    direction = np.array([forward_vec[0], forward_vec[2]])  # XZ→畫面XY
    draw_arrow(image, origin, direction, color=color, scale=scale, thickness=3)


# ───────────────────────────────────────────────────────────────
# analyzer.py 內容（內/外八分析）
_REQUIRED_KEYS = [
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
    "left_shoulder",
    "right_shoulder",
]

import mediapipe as mp
_MP_LM = mp.solutions.pose.PoseLandmark
_LM_INDEX = {lm.name.lower(): lm.value for lm in _MP_LM}


def _to_np(lm) -> np.ndarray:
    return (
        np.array([lm.x, lm.y, lm.z], dtype=float)
        if hasattr(lm, "x")
        else np.asarray(lm)
    )


def _ensure_dict(seq: Sequence[Any]) -> Dict[str, np.ndarray]:
    if not _LM_INDEX:
        raise ValueError("請提供 mediapipe 格式或 dict 格式")
    return {k: _to_np(seq[_LM_INDEX[k]]) for k in _REQUIRED_KEYS}


def _normalize_dict_keys(lm_dict: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    return {
        k.strip().lower().replace("pose_landmark.", ""): _to_np(v)
        for k, v in lm_dict.items()
    }


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def get_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0)
    return degrees(acos(dot))


def analyze_leg_rotation(
    landmarks: Any,
    previous: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:

    if isinstance(landmarks, Mapping):
        lm = _normalize_dict_keys(landmarks)
    else:
        lm = _ensure_dict(landmarks)

    th = {"high": 8.0, "low": 3.0, "ema": 0.2, "hip_rot": 15.0}
    if thresholds:
        th.update(thresholds)

    if previous is None:
        previous = {
            "left_leg": "Neutral",
            "right_leg": "Neutral",
            "ema_angle": {"left": None, "right": None},
            "bf_vec": None,
            "hip_trace": deque(maxlen=15),
        }

    # 估計身體前向（依髖部移動軌跡 XZ）
    hip_center = (lm["left_hip"] + lm["right_hip"]) / 2.0
    hip_xz = hip_center[[0, 2]]
    previous["hip_trace"].append(hip_xz)

    if len(previous["hip_trace"]) >= 2:
        delta = previous["hip_trace"][-1] - previous["hip_trace"][-2]
        if np.linalg.norm(delta) > 1e-3:
            body_forward = normalize(np.array([delta[0], 0.0, delta[1]]))
            previous["bf_vec"] = body_forward
        else:
            body_forward = (
                previous["bf_vec"]
                if previous["bf_vec"] is not None
                else np.array([1.0, 0.0, 0.0])
            )
    else:
        body_forward = np.array([1.0, 0.0, 0.0])

    result: Dict[str, Any] = {}
    for side in ("left", "right"):
        hip, knee = lm[f"{side}_hip"], lm[f"{side}_knee"]
        ankle, heel = lm[f"{side}_ankle"], lm[f"{side}_heel"]
        toe = lm[f"{side}_foot_index"]

        thigh_vec = knee - hip
        shank_vec = ankle - knee
        knee_angle = get_angle(thigh_vec, shank_vec)

        foot_vec_raw = toe - heel
        foot_proj = normalize(np.array([foot_vec_raw[0], 0.0, foot_vec_raw[2]]))

        cross_y = np.cross(body_forward, foot_proj)[1]
        outward = cross_y < 0 if side == "left" else cross_y > 0
        direction = "Out-Toed" if outward else "In-Toed"

        if np.linalg.norm(body_forward) < 1e-5 or np.linalg.norm(foot_proj) < 1e-5:
            raw_angle = previous["ema_angle"][side] or 0.0
        else:
            raw_angle = get_angle(foot_proj, body_forward)

        alpha = th["ema"]
        ema_prev = previous["ema_angle"][side]
        ema_angle = (
            raw_angle
            if ema_prev is None
            else alpha * raw_angle + (1 - alpha) * ema_prev
        )
        previous["ema_angle"][side] = ema_angle

        if previous.get(f"{side}_leg") == "In-Toed":
            label = "In-Toed" if ema_angle < th["low"] else "Neutral"
        elif previous.get(f"{side}_leg") == "Out-Toed":
            label = "Out-Toed" if ema_angle > th["high"] else "Neutral"
        else:
            if ema_angle < th["low"]:
                label = "In-Toed"
            elif ema_angle > th["high"]:
                label = "Out-Toed"
            else:
                label = "Neutral"

        thigh_proj = normalize(np.array([thigh_vec[0], 0.0, thigh_vec[2]]))
        hip_rot_angle = get_angle(thigh_proj, body_forward)
        hip_rot_status = (
            "Internally Rotated" if hip_rot_angle > th["hip_rot"] else "Neutral"
        )

        previous[f"{side}_leg"] = label

        result[f"{side}_leg"] = {
            "status": label,
            "direction": direction,
            "foot_angle_deg": round(ema_angle, 2),
            "knee_angle_deg": round(knee_angle, 2),
            "foot_proj_x": round(foot_proj[0], 4),
            "hip_rotation_deg": round(hip_rot_angle, 2),
            "hip_rotation_status": hip_rot_status,
        }

    return result, previous


# ───────────────────────────────────────────────────────────────
# pose_backend.py 內容（包 mediapipe + 視覺化）
class PoseAnalyzer:
    def __init__(self) -> None:
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._buf_left: deque[str] = deque(maxlen=60)
        self._buf_right: deque[str] = deque(maxlen=60)
        self._total_left: Counter[str] = Counter()
        self._total_right: Counter[str] = Counter()
        self._prev_labels: Optional[Dict[str, Any]] = None

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        img_h, img_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        rgb.flags.writeable = True

        analysis: Dict[str, Any] = {}
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            analysis, self._prev_labels = analyze_leg_rotation(
                lm, previous=self._prev_labels
            )

            # 左右腳箭頭
            for side in ("left", "right"):
                heel = lm[29 if side == "left" else 30]
                toe = lm[31 if side == "left" else 32]
                heel_pt = np.array([heel.x * img_w, heel.y * img_h])
                toe_pt = np.array([toe.x * img_w, toe.y * img_h])
                foot_vec = toe_pt - heel_pt
                foot_vec /= np.linalg.norm(foot_vec) + 1e-6
                draw_arrow(
                    frame_bgr,
                    heel_pt,
                    foot_vec,
                    color=(0, 255, 0) if side == "left" else (0, 0, 255),
                )

            # 身體前向箭頭（紫）
            hip_l, hip_r = lm[23], lm[24]
            hip_center = (
                (hip_l.x + hip_r.x) / 2 * img_w,
                (hip_l.y + hip_r.y) / 2 * img_h,
            )
            if self._prev_labels and "bf_vec" in self._prev_labels:
                bf = self._prev_labels["bf_vec"]
                if bf is not None:
                    draw_body_forward(frame_bgr, hip_center, bf)

            # 平滑狀態統計（簡易去抖）
            for side in ("left", "right"):
                buf = self._buf_left if side == "left" else self._buf_right
                buf.append(analysis[f"{side}_leg"]["status"])
                smoothed = max(set(buf), key=buf.count)
                (self._total_left if side == "left" else self._total_right)[
                    smoothed
                ] += 1

            # Mediapipe 骨架
            mp.solutions.drawing_utils.draw_landmarks(
                frame_bgr,
                result.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(245, 117, 66), thickness=2
                ),
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(245, 66, 230), thickness=2
                ),
            )

        return frame_bgr, analysis


# ───────────────────────────────────────────────────────────────
# gui_app.py 內容（PyQt 視窗 + 影像執行緒）
def build_metric_lines(left: dict, right: dict) -> list[str]:
    return [
        f"Left  status : {left.get('status', '--')}",
        f"Left  foot   : {left.get('foot_angle_deg', 0):.2f}",
        f"Left  hipRot : {left.get('hip_rotation_deg', 0):.2f}",
        "",
        f"Right status : {right.get('status', '--')}",
        f"Right foot   : {right.get('foot_angle_deg', 0):.2f}",
        f"Right hipRot : {right.get('hip_rotation_deg', 0):.2f}",
    ]


class VideoWorker(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(QtGui.QImage)
    analysisReady = QtCore.pyqtSignal(dict)

    def __init__(
        self,
        src=0,
        target_size=(608, 1080),
        output_path=None,
        draw_opencv_text=True,
        parent=None,
    ):
        super().__init__(parent)
        self._src = src
        self.tgt_w, self.tgt_h = target_size
        self.output_path = output_path
        self._backend = PoseAnalyzer()
        self._running = True
        self._draw_ocv_text = draw_opencv_text

    def run(self):
        cap = cv2.VideoCapture(self._src)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟來源：{self._src}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = None
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.output_path, fourcc, fps, (self.tgt_w, self.tgt_h)
            )

        while self._running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (self.tgt_w, self.tgt_h), cv2.INTER_AREA)
            frame, analysis = self._backend.process(frame)

            if self._draw_ocv_text and analysis:
                self._draw_metrics(frame, analysis)

            if writer is not None:
                writer.write(frame)

            self.analysisReady.emit(analysis if analysis else {})
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QtGui.QImage(
                rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888
            ).copy()
            self.frameReady.emit(qimg)

        if writer is not None:
            writer.release()
        cap.release()

    def _draw_metrics(self, frame: np.ndarray, analysis: dict) -> None:
        lines = build_metric_lines(
            analysis.get("left_leg", {}), analysis.get("right_leg", {})
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 1.0, 2
        txt_color = (200, 0, 0)
        bg_color = (255, 255, 255)
        pad = 8

        (w, h), _ = cv2.getTextSize(max(lines, key=len), font, scale, thick)
        line_gap = int(h * 1.35)
        bg_w = w + pad * 2
        bg_h = line_gap * len(lines) + pad * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (4, 4), (4 + bg_w, 4 + bg_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        y = 4 + pad + h
        for ln in lines:
            cv2.putText(
                frame, ln, (4 + pad, y), font, scale, txt_color, thick, cv2.LINE_AA
            )
            y += line_gap

    def stop(self):
        self._running = False
        self.wait()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, src, size, output, gui_text_only: bool):
        super().__init__()
        self.setWindowTitle("Foot Pose Analysis (PyQt)")
        self.videoLabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.last_analysis = {}

        self.worker = VideoWorker(
            src=src,
            target_size=size,
            output_path=output,
            draw_opencv_text=not gui_text_only,
        )
        self.worker.frameReady.connect(self.update_view)
        self.worker.analysisReady.connect(self.last_analysis.update)
        self.worker.start()

        self.setCentralWidget(self.videoLabel)

    @QtCore.pyqtSlot(QtGui.QImage)
    def update_view(self, img: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(img)
        painter = QtGui.QPainter(pix)
        painter.setFont(QtGui.QFont("Consolas", 16, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor(0, 0, 200))

        l = self.last_analysis.get("left_leg", {})
        r = self.last_analysis.get("right_leg", {})
        lines = build_metric_lines(l, r)

        fm = painter.fontMetrics()
        lh = fm.height() + 2
        pad = 6
        bg_w = max(fm.horizontalAdvance(t) for t in lines) + pad * 2
        bg_h = lh * len(lines) + pad * 2
        painter.fillRect(
            QtCore.QRect(0, 0, bg_w, bg_h), QtGui.QColor(255, 255, 255, 200)
        )

        x, y = pad, pad + fm.ascent()
        for txt in lines:
            painter.drawText(x, y, txt)
            y += lh

        painter.end()
        self.videoLabel.setPixmap(pix)

    def closeEvent(self, e):
        self.worker.stop()
        e.accept()


def _parse_size(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise ValueError("size 參數格式必須是 WxH，例如 608x1080")


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Foot Pose (all-in-one)")
    ap.add_argument(
        "--src",
        default="D:\\幼童足部辨識\\內八\\VID_20250630002612.mp4",
        help="影像來源：攝影機索引（如 0）或影片路徑。預設 0",
    )
    ap.add_argument("--size", default="608x1080", help="重採樣解析度，預設 608x1080")
    ap.add_argument(
        "--output", default="export/123.mp4", help="輸出 MP4 路徑（留空不輸出）"
    )
    ap.add_argument(
        "--gui-text-only", action="store_true", help="只用 GUI 疊字（關閉 OpenCV 疊字）"
    )
    args = ap.parse_args()

    src = 0 if args.src.strip() == "0" else args.src
    size = _parse_size(args.size)
    output = args.output or None

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(
        src=src, size=size, output=output, gui_text_only=args.gui_text_only
    )
    win.show()
    sys.exit(app.exec_())

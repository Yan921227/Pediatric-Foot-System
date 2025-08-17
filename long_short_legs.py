#!/usr/bin/env python
"""
PyQt5 visualizer – H‑K‑A 骨架線 + 關節角度疊加（字體加大／可調顏色／移除亂碼）
-----------------------------------------------------------------
• 即時顯示 L/R‑Hip、Knee、Ankle 角度。
• 若未指定 --video 會跳檔案對話框；若未指定 --out 自動存成 *_annotated.mp4。
"""

import sys, os, cv2, argparse
import numpy as np
import mediapipe as mp
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

# ── 外觀設定 ─────────────────────────────────────────
FONT          = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE    = 0.9         # 字體大小（0.5~1.5）
THICKNESS     = 2           # 字體粗細
BOX_COLOR     = (255, 255, 255)   # BGR，黑底
TEXT_COLOR    = (255, 0, 0)  # BGR，白字
BOX_ALPHA     = 1.0         # 透明度 (0~1)。1 為不透明，<1 需要額外混色，示例簡化不做。

# ───────────────────────────────────────────────────
def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """回傳 ∠ABC (deg)；輸入 2D 座標 ndarray"""
    ba, bc = a - b, c - b
    cos = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

# ───────────────────────────────────────────────────
class HKASkeletonViewer(QtWidgets.QMainWindow):
    def __init__(self, video_path: str, out_path: str | None):
        super().__init__()
        # 影片讀取 --------------------------------------------------------------
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resize(self.width, self.height)

        # 輸出影片 --------------------------------------------------------------
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = (cv2.VideoWriter(out_path, fourcc, self.fps,
                                       (self.width, self.height))
                       if out_path else None)

        # 顯示元件 --------------------------------------------------------------
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # MediaPipe Pose --------------------------------------------------------
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )

        # 更新迴圈 --------------------------------------------------------------
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.timer.start(int(1000 / self.fps))

    # -------------------------------------------------------------------------
    def _next_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self._finish(); return

        res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            angles = self._draw_hka(frame, res.pose_landmarks)
            self._overlay_angles(frame, angles)

        if self.writer:
            self.writer.write(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qt_img))

    # -------------------------------------------------------------------------
    def _draw_hka(self, frame, landmarks):
        h, w = frame.shape[:2]
        pts = [(lm.x * w, lm.y * h, lm.visibility) for lm in landmarks.landmark]
        pose = mp.solutions.pose.PoseLandmark
        idx = lambda name: pose[name].value
        L = {p: idx(f"LEFT_{p}")  for p in ("HIP", "KNEE", "ANKLE", "SHOULDER", "FOOT_INDEX")}
        R = {p: idx(f"RIGHT_{p}") for p in ("HIP", "KNEE", "ANKLE", "SHOULDER", "FOOT_INDEX")}

        angles = {}
        for side, I in (("L", L), ("R", R)):
            vis = [pts[i][2] for i in I.values()]
            if min(vis) < 0.6:
                angles[f"{side}_hip"] = angles[f"{side}_knee"] = angles[f"{side}_ankle"] = None
                continue

            hip, knee, ankle = [np.array(pts[I[p]][:2], int) for p in ("HIP", "KNEE", "ANKLE")]
            shd, foot        = [np.array(pts[I[p]][:2], int) for p in ("SHOULDER", "FOOT_INDEX")]

            cv2.line(frame, hip,  knee, (255, 255, 255), 2)
            cv2.line(frame, knee, ankle, (255, 255, 255), 2)

            cv2.circle(frame, hip,   8, (  0,165,255), -1)
            cv2.circle(frame, knee,  8, (  0,255,  0), -1)
            cv2.circle(frame, ankle, 8, (255,  0,  0), -1)

            angles[f"{side}_hip"]   = angle_between(shd, hip, knee)
            angles[f"{side}_knee"]  = angle_between(hip, knee, ankle)
            angles[f"{side}_ankle"] = angle_between(knee, ankle, foot)

        return angles

    # -------------------------------------------------------------------------
    def _overlay_angles(self, frame, angles):
        # ---- 準備字串（避免非 ASCII 符號）----
        lines = [
            f"L-Hip:   {angles['L_hip']:.2f}"   if angles['L_hip']   is not None else "L-Hip:   --",
            f"L-Knee:  {angles['L_knee']:.2f}"  if angles['L_knee']  is not None else "L-Knee:  --",
            f"L-Ankle: {angles['L_ankle']:.2f}" if angles['L_ankle'] is not None else "L-Ankle: --",
            f"R-Hip:   {angles['R_hip']:.2f}"   if angles['R_hip']   is not None else "R-Hip:   --",
            f"R-Knee:  {angles['R_knee']:.2f}"  if angles['R_knee']  is not None else "R-Knee:  --",
            f"R-Ankle: {angles['R_ankle']:.2f}" if angles['R_ankle'] is not None else "R-Ankle: --",
        ]

        # ---- 動態計算框尺寸 ----
        text_sizes = [cv2.getTextSize(t, FONT, FONT_SCALE, THICKNESS)[0] for t in lines]
        max_w = max(w for w, h in text_sizes)
        line_h = max(h for w, h in text_sizes) + 6  # 行高
        box_w  = max_w + 10
        box_h  = line_h * len(lines) + 4

        # ---- 畫框 ----
        cv2.rectangle(frame, (0, 0), (box_w, box_h), BOX_COLOR, -1)

        # ---- 寫字 ----
        y = line_h
        for ln in lines:
            cv2.putText(frame, ln, (5, y), FONT,
                        FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
            y += line_h

    # -------------------------------------------------------------------------
    def _finish(self):
        self.timer.stop()
        self.cap.release()
        if self.writer:
            self.writer.release()
        self.pose.close()
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！")
        QtWidgets.qApp.quit()

# ── CLI ───────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="H‑K‑A skeleton & angle visualizer")
    ap.add_argument("--video", help="影片路徑，空白會跳出檔案對話框")
    ap.add_argument("--out",   default="", help="輸出影片路徑；留空=自動 *_annotated.mp4；'none' 不存檔")
    args = ap.parse_args()

    # 影片路徑 ---------------------------------------------------------------
    if not args.video:
        app = QtWidgets.QApplication(sys.argv)  # 需先建 QApplication
        fname, _ = QFileDialog.getOpenFileName(None, "選擇影片", "", "Video Files (*.mp4 *.avi *.mov)")
        if not fname:
            print("未選擇影片，程式結束。"); return
        video_path = fname
    else:
        video_path = args.video.strip('"')

    # 輸出路徑 ---------------------------------------------------------------
    out_path = args.out.strip('"')
    if not out_path:
        root, _ = os.path.splitext(video_path)
        out_path = root + "_annotated.mp4"
    if out_path.lower() in ("none", "null", "no", "-"):
        out_path = None

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer = HKASkeletonViewer(video_path, out_path)
    viewer.setWindowTitle("HKA Skeleton Viewer")
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

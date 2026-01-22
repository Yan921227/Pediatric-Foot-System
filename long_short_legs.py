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
FONT_SCALE    = 2.0         # 字體大小(大幅增加)
THICKNESS     = 4           # 字體粗細(更粗更清楚)
BOX_COLOR     = (255, 255, 255)   # BGR,黑底
TEXT_COLOR    = (255, 0, 0)  # BGR,白字
BOX_ALPHA     = 1.0         # 透明度 (0~1)。1 為不透明,<1 需要額外混色,示例簡化不做。

# ── 長短腳判斷標準 ─────────────────────────────────────
STEP_LENGTH_DIFF_THRESHOLD = 5.0    # 步長差闾值(cm)
PELVIS_TILT_THRESHOLD = 5.0         # 骨盆傾斜闾值(度)
ASSUMED_HEIGHT_CM = 150.0           # 假設身高(cm)

# ───────────────────────────────────────────────────
def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """回傳 ∠ABC (deg);輸入 2D 座標 ndarray"""
    ba, bc = a - b, c - b
    cos = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """計算兩點間的2D距離"""
    return float(np.linalg.norm(p2 - p1))

def pixels_to_cm(dist_px: float, height_px: float, assumed_height_cm: float = ASSUMED_HEIGHT_CM) -> float:
    """將像素距離轉換為公分"""
    cm_per_px = assumed_height_cm / (height_px + 1e-6)
    return dist_px * cm_per_px

def estimate_height_px(hip: np.ndarray, ankle: np.ndarray) -> float:
    """估算身高(像素):從髖到踝的距離×2"""
    leg_length = distance_2d(hip, ankle)
    return leg_length * 2.0

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
        # 設定固定視窗尺寸，不允許縮放
        self.setFixedSize(self.width, self.height)

        # 輸出影片 --------------------------------------------------------------
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = (cv2.VideoWriter(out_path, fourcc, self.fps,
                                       (self.width, self.height))
                       if out_path else None)

        # 顯示元件 --------------------------------------------------------------
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        # 設定固定尺寸，不縮放圖片
        self.label.setFixedSize(self.width, self.height)
        self.label.setScaledContents(False)  # 不自動縮放內容
        self.setCentralWidget(self.label)

        # MediaPipe Pose --------------------------------------------------------
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )

        # 長短腳分析狀態 --------------------------------------------------------
        from collections import deque
        self.left_step_history = deque(maxlen=10)   # 左腳步長歷史
        self.right_step_history = deque(maxlen=10)  # 右腳步長歷史
        self.prev_left_ankle = None                 # 上一幀左踝位置
        self.prev_right_ankle = None                # 上一幀右踝位置
        self.lld_diagnosis = None                   # 長短腳診斷結果

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
            # 執行長短腳分析
            self.lld_diagnosis = self._analyze_leg_length_discrepancy(frame, res.pose_landmarks)
            # 疊加角度資訊
            self._overlay_angles(frame, angles)
            # 疊加長短腳診斷結果
            if self.lld_diagnosis:
                self._overlay_lld_diagnosis(frame, self.lld_diagnosis)

        if self.writer:
            self.writer.write(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        # 以原始尺寸顯示圖片，不進行縮放
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        self.label.setPixmap(pixmap)

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
        line_h = max(h for w, h in text_sizes) + 12  # 行高(適應更大字體)
        box_w  = max_w + 20
        box_h  = line_h * len(lines) + 12

        # ---- 畫框 ----
        cv2.rectangle(frame, (0, 0), (box_w, box_h), BOX_COLOR, -1)

        # ---- 寫字 ----
        y = line_h
        for ln in lines:
            cv2.putText(frame, ln, (12, y), FONT,
                        FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
            y += line_h

    # -------------------------------------------------------------------------
    def _analyze_leg_length_discrepancy(self, frame, landmarks):
        """分析長短腳:步長差、骨盆傾斜、壓力分析"""
        h, w = frame.shape[:2]
        pts = [(lm.x * w, lm.y * h, lm.visibility) for lm in landmarks.landmark]
        pose = mp.solutions.pose.PoseLandmark
        idx = lambda name: pose[name].value
        
        # 關鍵點索引
        L_HIP, R_HIP = idx("LEFT_HIP"), idx("RIGHT_HIP")
        L_ANKLE, R_ANKLE = idx("LEFT_ANKLE"), idx("RIGHT_ANKLE")
        
        # 檢查可見度
        if min(pts[L_HIP][2], pts[R_HIP][2], pts[L_ANKLE][2], pts[R_ANKLE][2]) < 0.6:
            return None
        
        # 提取座標
        l_hip = np.array(pts[L_HIP][:2], float)
        r_hip = np.array(pts[R_HIP][:2], float)
        l_ankle = np.array(pts[L_ANKLE][:2], float)
        r_ankle = np.array(pts[R_ANKLE][:2], float)
        
        # 1. 計算步長差(cm)
        step_length_diff_cm = self._calculate_step_length_diff(l_ankle, r_ankle, l_hip, r_hip)
        
        # 2. 計算骨盆傾斜(度)
        pelvis_tilt = self._calculate_pelvis_tilt(l_hip, r_hip)
        
        # 3. 綜合判斷(僅基於步長差和骨盆傾斜)
        diagnosis = {
            "step_diff_cm": step_length_diff_cm,
            "pelvis_tilt_deg": pelvis_tilt,
            "has_lld": False,
            "criteria_met": []
        }
        
        # 判斷標準:步長差 > 5cm 且 骨盆傾斜 > 5° (必須同時滿足)
        if step_length_diff_cm > STEP_LENGTH_DIFF_THRESHOLD and pelvis_tilt > PELVIS_TILT_THRESHOLD:
            diagnosis["has_lld"] = True
            diagnosis["criteria_met"].append("步長差+骨盆傾斜")
        
        return diagnosis
    
    def _calculate_step_length_diff(self, l_ankle, r_ankle, l_hip, r_hip):
        """計算步長差(cm)"""
        # 估算身高(取左右腿平均)
        l_height_px = estimate_height_px(l_hip, l_ankle)
        r_height_px = estimate_height_px(r_hip, r_ankle)
        avg_height_px = (l_height_px + r_height_px) / 2.0
        
        # 追蹤步長變化
        if self.prev_left_ankle is not None:
            l_step_px = distance_2d(l_ankle, self.prev_left_ankle)
            l_step_cm = pixels_to_cm(l_step_px, avg_height_px)
            if l_step_cm > 1.0:  # 過濾小移動
                self.left_step_history.append(l_step_cm)
        
        if self.prev_right_ankle is not None:
            r_step_px = distance_2d(r_ankle, self.prev_right_ankle)
            r_step_cm = pixels_to_cm(r_step_px, avg_height_px)
            if r_step_cm > 1.0:
                self.right_step_history.append(r_step_cm)
        
        self.prev_left_ankle = l_ankle.copy()
        self.prev_right_ankle = r_ankle.copy()
        
        # 計算平均步長差
        if len(self.left_step_history) >= 3 and len(self.right_step_history) >= 3:
            avg_l_step = np.mean(self.left_step_history)
            avg_r_step = np.mean(self.right_step_history)
            return abs(avg_l_step - avg_r_step)
        
        return 0.0
    
    def _calculate_pelvis_tilt(self, l_hip, r_hip):
        """計算骨盆傾斜角度(度)"""
        # 計算兩髖連線與水平線的夾角
        hip_vector = r_hip - l_hip
        horizontal = np.array([1.0, 0.0])
        
        # 計算夾角
        cos_angle = np.dot(hip_vector, horizontal) / (np.linalg.norm(hip_vector) * np.linalg.norm(horizontal) + 1e-6)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # 返回相對於水平的傾斜角
        return abs(90.0 - angle_deg)
    

    
    def _overlay_lld_diagnosis(self, frame, diagnosis):
        """疊加長短腳診斷結果"""
        if not diagnosis:
            return
        
        # 準備診斷文字
        lines = [
            "=== Long/Short Leg Diagnosis ===",
            f"Step diff: {diagnosis['step_diff_cm']:.2f} cm",
            f"Pelvis tilt: {diagnosis['pelvis_tilt_deg']:.2f} deg",
            ""
        ]
        
        # 診斷結果
        if diagnosis["has_lld"]:
            lines.append(f"Result: LLD Detected!")
            lines.append(f"Criteria: Step diff>5cm + Pelvis tilt>5deg")
            result_color = (0, 0, 255)  # 紅色警告
        else:
            lines.append("Result: Normal")
            result_color = (0, 255, 0)  # 綠色正常
        
        # 計算框尺寸
        text_sizes = [cv2.getTextSize(t, FONT, FONT_SCALE * 1.0, THICKNESS)[0] for t in lines]
        max_w = max(w for w, h in text_sizes)
        line_h = max(h for w, h in text_sizes) + 12
        box_w = max_w + 20
        box_h = line_h * len(lines) + 12
        
        # 位置:右上角
        x0 = frame.shape[1] - box_w - 10
        y0 = 10
        
        # 畫半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # 寫字
        y = y0 + line_h
        for i, ln in enumerate(lines):
            # 結果行用特殊顏色
            if "Result:" in ln:
                color = result_color
            else:
                color = TEXT_COLOR
            
            cv2.putText(frame, ln, (x0 + 12, y), FONT,
                        FONT_SCALE * 1.0, color, THICKNESS, cv2.LINE_AA)
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

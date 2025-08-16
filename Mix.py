#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動分類 → 分支執行：Tiptoe 或 HKA（長短腿角度）
------------------------------------------------
• 前置分類：讀取前 N 幀，判定 tiptoe vs. hka，僅執行其一。
• Tiptoe 模式：右上角 HUD（左右踝角度 + Gait）。
• HKA 模式：左上角面板（H-K-A 六行）。
• 字級：固定像素大小 --text-px（兩邊一致）；白底透明度與 HUD 一致。
• 未指定 --video 時，直接開「選擇影片」；輸出預設 <來源>_annotated.mp4。
相依：opencv-python, mediapipe, PyQt5, numpy
"""

import sys, os, cv2, argparse
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Optional
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

# ========================= 共同外觀設定 =========================
FONT            = cv2.FONT_HERSHEY_COMPLEX
THICKNESS       = 2   # 骨架線條粗細
TEXT_COLOR      = (255, 0, 0)  # 與 HUD 同色

# 面板（Tiptoe HUD 與 HKA 共用）
PANEL_W_RATIO   = 0.50    # 只用在 HUD；HKA 面板自適應寬度
PANEL_ALPHA     = 0.75    # ★ 白底透明度（兩邊一致）
PANEL_BG_COLOR  = (255, 255, 255)
HUD_TEXT_COLOR  = (255, 0, 0)
HUD_TEXT_THICK  = 1       # ★ 文字粗細（兩邊一致）

# 骨架點/線（視覺）
DOT_RADIUS      = 4
DOT_COLOR       = (0, 200, 0)
SEG_COLOR       = (180, 180, 180)

# ========================= 偵測/濾波參數（Tiptoe 亦用於分類） =====================
PF_THRESHOLD_DEG = 8.0      # 足底屈門檻（踝角）
SMA_WIN          = 7
EMA2D_ALPHA      = 0.35
EMA3D_ALPHA      = 0.35
ANGLE_DEF        = 'KAT'    # 'KAT' 或 'HAT'

# 分類參數
CLS_MAX_FRAMES   = 180      # 前置分類最多檢查幀數
CLS_MIN_VALID    = 60       # 至少有效幀
TIPTOE_RATIO_TH  = 0.15     # tiptoe 幀比例門檻

# ======= 字級：固定像素高度（不要跟影像大小聯動） =======
DEFAULT_TEXT_PX  = 10  # 可用 --text-px 覆蓋

def fixed_font_metrics(target_px: int, thickness: int = HUD_TEXT_THICK, sample_text: str = "Hg"):
    """依固定像素高度換算 OpenCV 的 fontScale，並回傳 (font_scale, 行高, 內距)。"""
    target_px = max(8, int(target_px))
    ((_, base_h), _) = cv2.getTextSize(sample_text, FONT, 1.0, thickness)
    font_scale = float(target_px) / max(1, base_h)
    line_h = int(target_px + max(4, target_px * 0.2))  # 行高：比字高多一點間距
    pad    = int(max(8, target_px * 0.6))              # 面板內距
    return font_scale, line_h, pad

# ========================= 小工具 =========================
class SMA:
    def __init__(self, win=7): self.q = deque(maxlen=win)
    def push(self, v): self.q.append(float(v));  return sum(self.q)/len(self.q)

class EMA:
    def __init__(self, alpha=0.3): self.alpha=float(alpha); self.v=None
    def push(self, x):
        x = np.asarray(x, dtype=float)
        self.v = x if self.v is None else self.alpha*x + (1-self.alpha)*self.v
        return self.v

def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cos = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def angle_3pt(a, b, c):
    a,b,c = map(np.asarray,(a,b,c))
    v1, v2 = a-b, c-b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosv  = float(np.dot(v1, v2) / denom)
    return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))

def fmt_num(val, digits=2):
    if val is None or np.isnan(val) or np.isinf(val):  return "N/A"
    return f"{float(val):.{digits}f}"

# ========================= 前置分類 =========================
def classify_video(video_path: str) -> str:
    """
    回傳 'tiptoe' 或 'hka'
    以前 CLS_MAX_FRAMES 幀中 (toe 高於 heel 且 踝角>門檻) 的比例作為依據。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.6,
        smooth_landmarks=True
    )

    valid = 0
    tiptoe_frames = 0

    ema2d = {i: EMA(EMA2D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}
    ema3d = {i: EMA(EMA3D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}
    smaL, smaR = SMA(SMA_WIN), SMA(SMA_WIN)

    for _ in range(CLS_MAX_FRAMES):
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks: continue

        lm2d = res.pose_landmarks.landmark
        world = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None

        def get2d(i):
            u = lm2d[i]; return ema2d[i].push(np.array([u.x*w, u.y*h], float))
        def get3d(i):
            if world is None:
                x,y = get2d(i); pt = np.array([x, y, 0.0], float)
            else:
                u = world[i];   pt = np.array([u.x, u.y, u.z], float)
            return ema3d[i].push(pt)

        L_KNEE, L_ANK, L_HEEL, L_TOE = 25, 27, 29, 31
        R_KNEE, R_ANK, R_HEEL, R_TOE = 26, 28, 30, 32

        if ANGLE_DEF.upper() == 'KAT':
            L_raw = angle_3pt(get3d(L_KNEE), get3d(L_ANK), get3d(L_TOE))
            R_raw = angle_3pt(get3d(R_KNEE), get3d(R_ANK), get3d(R_TOE))
        else:
            L_raw = angle_3pt(get3d(L_HEEL), get3d(L_ANK), get3d(L_TOE))
            R_raw = angle_3pt(get3d(R_HEEL), get3d(R_ANK), get3d(R_TOE))

        L_pf = smaL.push(L_raw)
        R_pf = smaR.push(R_raw)

        left_tiptoe  = (L_pf > PF_THRESHOLD_DEG) and (lm2d[L_TOE].y < lm2d[L_HEEL].y)
        right_tiptoe = (R_pf > PF_THRESHOLD_DEG) and (lm2d[R_TOE].y < lm2d[R_HEEL].y)

        valid += 1
        if left_tiptoe or right_tiptoe:
            tiptoe_frames += 1

    cap.release(); pose.close()

    if valid < max(CLS_MIN_VALID, int(0.3 * CLS_MAX_FRAMES)):
        print(f"[Classify] valid={valid} < minimum, fallback=hka")
        return "hka"

    ratio = (tiptoe_frames / float(valid))
    label = "tiptoe" if ratio >= TIPTOE_RATIO_TH else "hka"
    print(f"[Classify] valid={valid}, tiptoe_frames={tiptoe_frames}, ratio={ratio:.3f} => {label}")
    return label

# ========================= Tiptoe Viewer（右上角 HUD） =========================
class TiptoeViewer(QtWidgets.QMainWindow):
    def __init__(self, video_path: str, out_path: Optional[str], text_px: int):
        super().__init__()
        self.text_px = max(8, int(text_px))

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resize(self.width, self.height)

        self.writer = None
        if out_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.width, self.height))

        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.label)

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=2,
            min_detection_confidence=0.6, min_tracking_confidence=0.6,
            smooth_landmarks=True
        )

        self.frame_count = 0
        self.tiptoe_frames = 0

        self.ema2d = {i: EMA(EMA2D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}
        self.ema3d = {i: EMA(EMA3D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}
        self.sma_pf = {"L": SMA(SMA_WIN), "R": SMA(SMA_WIN)}

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.timer.start(int(1000 / self.fps))

    def _next_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self._finish(); return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        L_show = R_show = None
        gait_txt = "Normal"

        if res.pose_landmarks:
            lm2d = res.pose_landmarks.landmark
            world = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None

            def get2d(i):
                u = lm2d[i]; return self.ema2d[i].push(np.array([u.x*self.width, u.y*self.height], float))
            def get3d(i):
                if world is None:
                    x,y = get2d(i); pt = np.array([x, y, 0.0], float)
                else:
                    u = world[i];   pt = np.array([u.x, u.y, u.z], float)
                return self.ema3d[i].push(pt)

            L_KNEE, L_ANK, L_HEEL, L_TOE = 25, 27, 29, 31
            R_KNEE, R_ANK, R_HEEL, R_TOE = 26, 28, 30, 32

            if ANGLE_DEF.upper() == 'KAT':
                L_raw = angle_3pt(get3d(L_KNEE), get3d(L_ANK), get3d(L_TOE))
                R_raw = angle_3pt(get3d(R_KNEE), get3d(R_ANK), get3d(R_TOE))
            else:
                L_raw = angle_3pt(get3d(L_HEEL), get3d(L_ANK), get3d(L_TOE))
                R_raw = angle_3pt(get3d(R_HEEL), get3d(R_ANK), get3d(R_TOE))

            L_pf = self.sma_pf["L"].push(L_raw)
            R_pf = self.sma_pf["R"].push(R_raw)

            left_tiptoe  = (L_pf > PF_THRESHOLD_DEG) and (lm2d[L_TOE].y < lm2d[L_HEEL].y)
            right_tiptoe = (R_pf > PF_THRESHOLD_DEG) and (lm2d[R_TOE].y < lm2d[R_HEEL].y)

            self._draw_side_min(frame, get2d, L_KNEE, L_ANK, L_TOE, L_HEEL)
            self._draw_side_min(frame, get2d, R_KNEE, R_ANK, R_TOE, R_HEEL)

            if left_tiptoe:  L_show = float(L_pf)
            if right_tiptoe: R_show = float(R_pf)
            if left_tiptoe or right_tiptoe:
                gait_txt = "Tiptoe"
                self.tiptoe_frames += 1

        # —— 右上角 HUD（固定像素字級） ——
        fs, line_h, pad = fixed_font_metrics(self.text_px)
        lines = [
            f"Left ankle angle:  {fmt_num(L_show)} deg",
            f"Right ankle angle: {fmt_num(R_show)} deg",
            f"Gait: {gait_txt}"
        ]
        self._draw_info_panel_top_right(frame, lines, fs, line_h, pad)

        if self.writer: self.writer.write(frame)
        qimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_BGR888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self.frame_count += 1

    def _draw_info_panel_top_right(self, frame, lines, font_scale, line_h, pad):
        h, w = frame.shape[:2]
        panel_w = int(w * PANEL_W_RATIO)
        box_h   = pad*2 + line_h*len(lines)
        x0, y0  = w - panel_w - 10, 10  # 右上
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+box_h), PANEL_BG_COLOR, -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1-PANEL_ALPHA, 0, frame)
        y = y0 + pad + line_h - int(line_h*0.3)
        for text in lines:
            cv2.putText(frame, text, (x0+pad, y), FONT, font_scale, HUD_TEXT_COLOR, HUD_TEXT_THICK, cv2.LINE_AA)
            y += line_h

    def _draw_side_min(self, frame, get2d, K, A, T, H):
        xK,yK = map(int, get2d(K)); xA,yA = map(int, get2d(A))
        xT,yT = map(int, get2d(T)); xH,yH = map(int, get2d(H))
        cv2.circle(frame,(xK,yK),DOT_RADIUS,DOT_COLOR,-1)
        cv2.circle(frame,(xA,yA),DOT_RADIUS,DOT_COLOR,-1)
        cv2.circle(frame,(xT,yT),DOT_RADIUS,DOT_COLOR,-1)
        cv2.circle(frame,(xH,yH),DOT_RADIUS,DOT_COLOR,-1)
        cv2.line(frame,(xK,yK),(xA,yA),SEG_COLOR,2)
        cv2.line(frame,(xA,yA),(xT,yT),SEG_COLOR,2)

    def _finish(self):
        self.timer.stop(); self.cap.release()
        if self.writer: self.writer.release()
        self.pose.close()
        if self.frame_count > 0:
            ratio = self.tiptoe_frames / self.frame_count
            label = "Tiptoe" if ratio >= TIPTOE_RATIO_TH else "Normal"
            print(f"[Summary] frames={self.frame_count}, tiptoe_frames={self.tiptoe_frames}, ratio={ratio:.3f}, video_gait={label}")
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！"); QtWidgets.qApp.quit()

# ========================= HKA Viewer（左上角面板；字級與透明度與 HUD 一致） =========================
class HKAViewer(QtWidgets.QMainWindow):
    def __init__(self, video_path: str, out_path: Optional[str], text_px: int):
        super().__init__()
        self.text_px = max(8, int(text_px))

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resize(self.width, self.height)

        self.writer = None
        if out_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.width, self.height))

        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.label)

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=2,
            min_detection_confidence=0.6, min_tracking_confidence=0.6,
            smooth_landmarks=True
        )

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.timer.start(int(1000 / self.fps))

    def _next_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self._finish(); return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if res.pose_landmarks:
            angles = self._compute_hka_angles(frame, res.pose_landmarks)
            fs, line_h, pad = fixed_font_metrics(self.text_px)  # ★ 固定像素字級
            self._overlay_angles_top_left(frame, angles, fs, line_h, pad)

        if self.writer: self.writer.write(frame)
        qimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_BGR888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _compute_hka_angles(self, frame, landmarks):
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

    def _overlay_angles_top_left(self, frame, angles, font_scale, line_h, pad):
        lines = [
            f"L-Hip:   {angles['L_hip']:.2f}"   if angles['L_hip']   is not None else "L-Hip:   --",
            f"L-Knee:  {angles['L_knee']:.2f}"  if angles['L_knee']  is not None else "L-Knee:  --",
            f"L-Ankle: {angles['L_ankle']:.2f}" if angles['L_ankle'] is not None else "L-Ankle: --",
            f"R-Hip:   {angles['R_hip']:.2f}"   if angles['R_hip']   is not None else "R-Hip:   --",
            f"R-Knee:  {angles['R_knee']:.2f}"  if angles['R_knee']  is not None else "R-Knee:  --",
            f"R-Ankle: {angles['R_ankle']:.2f}" if angles['R_ankle'] is not None else "R-Ankle: --",
        ]
        # 面板寬度（自適應文字寬）
        text_sizes = [cv2.getTextSize(t, FONT, font_scale, HUD_TEXT_THICK)[0] for t in lines]
        max_w = max(w for w, h in text_sizes)
        box_w = max_w + pad*2
        box_h = pad*2 + line_h*len(lines)

        # —— 左上角半透明白底（與 HUD 同法、同透明度） ——
        x0, y0 = 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+box_w, y0+box_h), PANEL_BG_COLOR, -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1-PANEL_ALPHA, 0, frame)

        # 寫字（同色、同粗細）
        y = y0 + pad + line_h - int(line_h*0.3)
        for ln in lines:
            cv2.putText(frame, ln, (x0+pad, y), FONT, font_scale, HUD_TEXT_COLOR, HUD_TEXT_THICK, cv2.LINE_AA)
            y += line_h

    def _finish(self):
        self.timer.stop(); self.cap.release()
        if self.writer: self.writer.release()
        self.pose.close()
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！"); QtWidgets.qApp.quit()

# ========================= 主入口 =========================
def main():
    ap = argparse.ArgumentParser(description="Auto Gait Router: tiptoe 或 hka（長短腿角度）")
    ap.add_argument("--video", help="影片路徑；空白會直接開檔案選擇視窗")
    ap.add_argument("--out",   default="", help="輸出影片；留空=自動 *_annotated.mp4；'none' 不存檔")
    ap.add_argument("--force", choices=["auto","tiptoe","hka"], default="auto",
                    help="強制模式（預設 auto：先分類再分支）")
    ap.add_argument("--text-px", type=int, default=DEFAULT_TEXT_PX,
                    help="固定字高（像素），預設 30")
    args = ap.parse_args()

    # 建立/取得 QApplication（為了檔案選擇與 GUI）
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # 影片路徑
    if not args.video:
        fname, _ = QFileDialog.getOpenFileName(None, "選擇影片", "", "Video Files (*.mp4 *.avi *.mov)")
        if not fname:
            print("未選擇影片，程式結束。"); return
        video_path = fname
    else:
        video_path = args.video.strip('"')

    # 輸出路徑
    out_path = args.out.strip('"')
    if not out_path:
        root, _ = os.path.splitext(video_path)
        out_path = root + "_annotated.mp4"
    if out_path.lower() in ("none", "null", "no", "-"):
        out_path = None

    # 分類或強制
    mode = args.force
    if mode == "auto":
        mode = classify_video(video_path)  # 'tiptoe' or 'hka'

    # 啟動對應 Viewer（只開一種），字級用參數傳入
    text_px = max(8, int(args.text_px))
    if mode == "tiptoe":
        viewer = TiptoeViewer(video_path, out_path, text_px=text_px)
        viewer.setWindowTitle("Gait Analyzer – Tiptoe")
    else:
        viewer = HKAViewer(video_path, out_path, text_px=text_px)
        viewer.setWindowTitle("Gait Analyzer – HKA")
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

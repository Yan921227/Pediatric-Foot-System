#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto Gait Router：Tiptoe / HKA（長短腳角度）/ In-Out-Toeing（內外八）
-------------------------------------------------------------------
• 不用命令列調參：內部已設定保守、穩定的固定參數。
• 前置分類（自動分流）：讀取較長時間 + 暖機，降低誤分。
• 三種模式只會顯示其中一種（不混顯）。
    1) Tiptoe（踮腳）：右上角 HUD。
    2) HKA（髖膝踝六行）：左上角面板。
    3) In-Out-Toeing（內/外八 + 髖內旋）：左上角面板（含骨架）。

• 疊字與樣式：
    - 全部使用 OpenCV 疊字；字級「固定像素」（DEFAULT_TEXT_PX）。
    - 面板白底透明度 PANEL_ALPHA；三模式一致。
    - In/Out 會畫全身骨架與箭頭。

• 輸入：未指定 --video 時會開檔案選擇器。
• 輸出：未指定 --out → <來源>_annotated.mp4；--out none 不存檔。

依賴：opencv-python, mediapipe, PyQt5, numpy
"""

import sys, os, cv2, argparse
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Optional, Any, Dict, Tuple
from math import acos, degrees
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

# ========================= 外觀設定 =========================
FONT            = cv2.FONT_HERSHEY_COMPLEX
HUD_TEXT_COLOR  = (255, 0, 0)
HUD_TEXT_THICK  = 1
PANEL_BG_COLOR  = (255, 255, 255)
PANEL_ALPHA     = 0.75     # 三模式一致
PANEL_W_RATIO   = 0.50     # 只用在 Tiptoe 右上 HUD
SEG_COLOR       = (180, 180, 180)
DOT_COLOR       = (0, 200, 0)
DOT_RADIUS      = 4

# ======= 字級：固定像素（不依影像大小伸縮） =======
DEFAULT_TEXT_PX = 10
def fixed_font_metrics(target_px: int, thickness: int = HUD_TEXT_THICK, sample_text: str = "Hg"):
    target_px = max(8, int(target_px))
    ((_, base_h), _) = cv2.getTextSize(sample_text, FONT, 1.0, thickness)
    font_scale = float(target_px) / max(1, base_h)
    line_h = int(target_px + max(4, target_px * 0.2))
    pad    = int(max(8, target_px * 0.6))
    return font_scale, line_h, pad

# ========================= 穩定預設（Tiptoe 也供分類用） =========================
# Tiptoe 平滑（保守、穩定）
PF_THRESHOLD_DEG = 10.0      # 踝角門檻（較嚴格）
SMA_WIN          = 11        # 踝角 SMA 視窗（越大越穩）
EMA2D_ALPHA      = 0.18      # 關鍵點 2D EMA（越小越穩）
EMA3D_ALPHA      = 0.18      # 關鍵點 3D EMA

ANGLE_DEF        = 'KAT'     # 踝角定義：'KAT' 或 'HAT'

# HKA 也加平滑（原本沒有）：針對 2D 關鍵點再做 EMA
HKA_EMA_ALPHA    = 0.22
HKA_SMA_WIN      = 9         # 六個角度再做 SMA，讓數字更穩

# ========================= 自動分流參數（保守） =========================
CLS_MAX_FRAMES  = 900        # 觀察更久
CLS_MIN_VALID   = 250        # 最少有效幀
TIPTOE_RATIO_TH = 0.25       # 要至少 25% 幀踮腳才分到 tiptoe
INOUT_RATIO_TH  = 0.55       # 要至少 55% 幀出現內/外八才分到 inout
BURN_IN         = 90         # 暖機幀：前 90 幀不計入比例

# ========================= 小工具（SMA/EMA/角度） =========================
class SMA:
    def __init__(self, win=7): self.q = deque(maxlen=win)
    def push(self, v): self.q.append(float(v));  return sum(self.q)/len(self.q)

class EMA:
    def __init__(self, alpha=0.3): self.alpha=float(alpha); self.v=None
    def push(self, x):
        x = np.asarray(x, dtype=float)
        self.v = x if self.v is None else self.alpha*x + (1-self.alpha)*self.v
        return self.v

def angle_3pt(a, b, c):
    a,b,c = map(np.asarray,(a,b,c))
    v1, v2 = a-b, c-b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosv  = float(np.dot(v1, v2) / denom)
    return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))

def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cos = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

# ========================= In/Out-Toeing 工具與分析（內建更穩的門檻） =========================
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v);  return v if n == 0 else v / n

def _ang(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.clip(np.dot(_normalize(v1), _normalize(v2)), -1.0, 1.0)
    return degrees(acos(dot))

def draw_arrow(image, origin_xy, vec_xy, color=(0,255,0), scale=100, thickness=2):
    p1 = tuple(np.round(origin_xy).astype(int))
    p2 = tuple(np.round(origin_xy + vec_xy * scale).astype(int))
    cv2.arrowedLine(image, p1, p2, color, thickness, tipLength=0.3)

def draw_body_forward(image, center_xy, forward_vec_xz, scale=100, color=(255,0,255)):
    vec_xy = np.array([forward_vec_xz[0], forward_vec_xz[2]])
    draw_arrow(image, np.array(center_xy), vec_xy, color=color, scale=scale, thickness=3)

def analyze_inout_frame(lm, prev: Optional[Dict[str,Any]] = None,
                        th: Optional[Dict[str,float]] = None) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    # 內建更穩的設定：較小 EMA、較大的遲滯
    if th is None:
        th = {"high": 12.0, "low": 3.0, "ema": 0.18, "hip_rot": 15.0}
    if prev is None:
        prev = {"left_leg":"Neutral","right_leg":"Neutral",
                "ema_angle":{"left":None,"right":None},
                "bf_vec":None,"hip_trace":deque(maxlen=15)}

    # 估計身體前向（依髖中心在 XZ 的位移）
    hip_l, hip_r = lm[23], lm[24]
    hip_center = np.array([(hip_l.x+hip_r.x)/2.0, (hip_l.y+hip_r.y)/2.0, (hip_l.z+hip_r.z)/2.0], float)
    hip_xz = hip_center[[0,2]]
    prev["hip_trace"].append(hip_xz)
    if len(prev["hip_trace"]) >= 2:
        delta = prev["hip_trace"][-1] - prev["hip_trace"][-2]
        if np.linalg.norm(delta) > 1e-3:
            bf = _normalize(np.array([delta[0], 0.0, delta[1]], float))
            prev["bf_vec"] = bf
        else:
            bf = prev["bf_vec"] if prev["bf_vec"] is not None else np.array([1.0,0.0,0.0])
    else:
        bf = np.array([1.0,0.0,0.0])

    result = {}
    for side, idxs in (("left",(29,31)), ("right",(30,32))):  # heel, toe
        heel = lm[idxs[0]]; toe = lm[idxs[1]]
        foot_vec_raw = np.array([toe.x-heel.x, toe.y-heel.y, toe.z-heel.z], float)
        foot_proj = _normalize(np.array([foot_vec_raw[0], 0.0, foot_vec_raw[2]]))
        raw = prev["ema_angle"][side] or 0.0 if (np.linalg.norm(bf)<1e-5 or np.linalg.norm(foot_proj)<1e-5) else _ang(foot_proj, bf)

        # EMA（更小 α → 更穩）
        alpha = th["ema"]; prev_val = prev["ema_angle"][side]
        ema = raw if prev_val is None else alpha*raw + (1-alpha)*prev_val
        prev["ema_angle"][side] = ema

        # 三態（遲滯：不容易來回跳）
        if prev.get(f"{side}_leg") == "In-Toed":
            label = "In-Toed" if ema < th["low"] else "Neutral"
        elif prev.get(f"{side}_leg") == "Out-Toed":
            label = "Out-Toed" if ema > th["high"] else "Neutral"
        else:
            label = "In-Toed" if ema < th["low"] else ("Out-Toed" if ema > th["high"] else "Neutral")
        prev[f"{side}_leg"] = label

        # 髖內旋估計
        knee = lm[25 if side=="left" else 26]
        hip  = lm[23 if side=="left" else 24]
        thigh = np.array([knee.x-hip.x, knee.y-hip.y, knee.z-hip.z], float)
        thigh_proj = _normalize(np.array([thigh[0], 0.0, thigh[2]]))
        hip_rot_angle = _ang(thigh_proj, bf)
        hip_rot_status = "Internally Rotated" if hip_rot_angle > th["hip_rot"] else "Neutral"

        result[f"{side}_leg"] = {
            "status": label,
            "foot_angle_deg": round(ema, 2),
            "hip_rotation_deg": round(hip_rot_angle, 2),
            "hip_rotation_status": hip_rot_status,
            "heel_xy": (heel.x, heel.y),
            "toe_xy":  (toe.x,  toe.y),
        }
    result["bf_vec"] = bf
    result["hip_center_xy"] = ((hip_l.x+hip_r.x)/2.0, (hip_l.y+hip_r.y)/2.0)
    return result, prev

def build_inout_lines(L:dict, R:dict) -> list[str]:
    return [
        f"Left  status : {L.get('status','--')}",
        f"Left  foot   : {L.get('foot_angle_deg',0):.2f}",
        f"Left  hipRot : {L.get('hip_rotation_deg',0):.2f}",
        "",
        f"Right status : {R.get('status','--')}",
        f"Right foot   : {R.get('foot_angle_deg',0):.2f}",
        f"Right hipRot : {R.get('hip_rotation_deg',0):.2f}",
    ]

# ========================= 自動分流（Tiptoe / InOut / HKA） =========================
def classify_video(video_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.6,
        smooth_landmarks=True
    )
    ema2d = {i: EMA(EMA2D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}
    ema3d = {i: EMA(EMA3D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}
    smaL, smaR = SMA(SMA_WIN), SMA(SMA_WIN)

    valid = 0
    tiptoe_frames = 0
    inout_frames  = 0
    inout_prev = None

    for i in range(CLS_MAX_FRAMES):
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            continue

        lm2d = res.pose_landmarks.landmark
        world = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None

        def get2d(i2):
            u = lm2d[i2]; return ema2d[i2].push(np.array([u.x*w, u.y*h], float))
        def get3d(i2):
            if world is None:
                x,y = get2d(i2); pt = np.array([x, y, 0.0], float)
            else:
                u = world[i2];   pt = np.array([u.x, u.y, u.z], float)
            return ema3d[i2].push(pt)

        # Tiptoe 指標
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

        # In/Out 指標
        analysis, inout_prev = analyze_inout_frame(lm2d, prev=inout_prev)
        any_inout = (analysis["left_leg"]["status"]  != "Neutral") or \
                    (analysis["right_leg"]["status"] != "Neutral")

        # 暖機幀不記入比例
        if i < BURN_IN:
            continue

        valid += 1
        if left_tiptoe or right_tiptoe: tiptoe_frames += 1
        if any_inout: inout_frames += 1

    cap.release(); pose.close()

    if valid < max(CLS_MIN_VALID, int(0.3 * CLS_MAX_FRAMES)):
        print(f"[Classify] valid={valid} < minimum, fallback=hka")
        return "hka"

    tiptoe_ratio = tiptoe_frames / float(valid)
    inout_ratio  = inout_frames  / float(valid)
    print(f"[Classify] valid={valid}, tiptoe={tiptoe_frames} ({tiptoe_ratio:.3f}), inout={inout_frames} ({inout_ratio:.3f})")

    if tiptoe_ratio >= TIPTOE_RATIO_TH: return "tiptoe"
    if inout_ratio  >= INOUT_RATIO_TH:  return "inout"
    return "hka"

# ========================= Tiptoe Viewer（右上 HUD） =========================
class TiptoeViewer(QtWidgets.QMainWindow):
    def __init__(self, video_path: str, out_path: Optional[str], text_px: int):
        super().__init__()
        self.text_px = max(8, int(text_px))
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): raise RuntimeError(f"無法開啟影片：{video_path}")
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
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2,
                                           min_detection_confidence=0.6, min_tracking_confidence=0.6,
                                           smooth_landmarks=True)
        self.frame_count = 0; self.tiptoe_frames = 0
        self.ema2d = {i: EMA(EMA2D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}
        self.ema3d = {i: EMA(EMA3D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}
        self.sma_pf = {"L": SMA(SMA_WIN), "R": SMA(SMA_WIN)}
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._next_frame)
        self.timer.start(int(1000 / self.fps))

    def _next_frame(self):
        ok, frame = self.cap.read()
        if not ok: self._finish(); return
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
            L_pf = self.sma_pf["L"].push(L_raw);  R_pf = self.sma_pf["R"].push(R_raw)
            left_tiptoe  = (L_pf > PF_THRESHOLD_DEG) and (lm2d[L_TOE].y < lm2d[L_HEEL].y)
            right_tiptoe = (R_pf > PF_THRESHOLD_DEG) and (lm2d[R_TOE].y < lm2d[R_HEEL].y)
            # 下肢簡化關聯線
            self._draw_side_min(frame, lm2d, 25, 27, 31, 29); self._draw_side_min(frame, lm2d, 26, 28, 32, 30)
            if left_tiptoe:  L_show = float(L_pf)
            if right_tiptoe: R_show = float(R_pf)
            if left_tiptoe or right_tiptoe:
                gait_txt = "Tiptoe"; self.tiptoe_frames += 1
        # 右上 HUD
        fs, line_h, pad = fixed_font_metrics(self.text_px)
        lines = [f"Left ankle angle:  {('N/A' if L_show is None else f'{L_show:.2f}')} deg",
                 f"Right ankle angle: {('N/A' if R_show is None else f'{R_show:.2f}')} deg",
                 f"Gait: {gait_txt}"]
        self._draw_info_panel_top_right(frame, lines, fs, line_h, pad)
        if self.writer: self.writer.write(frame)
        qimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_BGR888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg)); self.frame_count += 1

    def _draw_info_panel_top_right(self, frame, lines, font_scale, line_h, pad):
        h, w = frame.shape[:2]; panel_w = int(w * PANEL_W_RATIO);  box_h = pad*2 + line_h*len(lines)
        x0, y0  = w - panel_w - 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+box_h), PANEL_BG_COLOR, -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1-PANEL_ALPHA, 0, frame)
        y = y0 + pad + line_h - int(line_h*0.3)
        for text in lines:
            cv2.putText(frame, text, (x0+pad, y), FONT, font_scale, HUD_TEXT_COLOR, HUD_TEXT_THICK, cv2.LINE_AA)
            y += line_h

    def _draw_side_min(self, frame, lm2d, K, A, T, H):
        h, w = frame.shape[:2]
        pts = lambda i: (int(lm2d[i].x*w), int(lm2d[i].y*h))
        xK,yK = pts(K); xA,yA = pts(A); xT,yT = pts(T); xH,yH = pts(H)
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
        if self.frame_count>0:
            ratio = self.tiptoe_frames / self.frame_count
            label = "Tiptoe" if ratio >= TIPTOE_RATIO_TH else "Normal"
            print(f"[Summary] frames={self.frame_count}, tiptoe_frames={self.tiptoe_frames}, ratio={ratio:.3f}, video_gait={label}")
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！"); QtWidgets.qApp.quit()

# ========================= HKA Viewer（左上面板；已加平滑） =========================
class HKAViewer(QtWidgets.QMainWindow):
    def __init__(self, video_path: str, out_path: Optional[str], text_px: int):
        super().__init__()
        self.text_px = max(8, int(text_px))
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): raise RuntimeError(f"無法開啟影片：{video_path}")
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resize(self.width, self.height)
        self.writer = None
        if out_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.width, self.height))
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter); self.setCentralWidget(self.label)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2,
                                           min_detection_confidence=0.6, min_tracking_confidence=0.6,
                                           smooth_landmarks=True)
        # ★ HKA 也做 2D 關鍵點 EMA + 角度 SMA
        self.hka_ema = {i: EMA(HKA_EMA_ALPHA) for i in (
            mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
            mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value,
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
            mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
        )}
        keys = ["L_hip","L_knee","L_ankle","R_hip","R_knee","R_ankle"]
        self.hka_sma = {k: SMA(HKA_SMA_WIN) for k in keys}

        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._next_frame)
        self.timer.start(int(1000 / self.fps))

    def _next_frame(self):
        ok, frame = self.cap.read()
        if not ok: self._finish(); return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            angles = self._compute_hka_angles(frame, res.pose_landmarks)
            fs, line_h, pad = fixed_font_metrics(self.text_px)
            self._overlay_angles_top_left(frame, angles, fs, line_h, pad)
        if self.writer: self.writer.write(frame)
        qimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_BGR888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _get_pt(self, lm, idx, w, h):
        u = lm[idx]
        p = self.hka_ema[idx].push(np.array([u.x*w, u.y*h], float))
        return np.array(p, int)

    def _compute_hka_angles(self, frame, landmarks):
        h, w = frame.shape[:2]
        lm = landmarks.landmark
        P = mp.solutions.pose.PoseLandmark
        # 取平滑後的 2D 點
        LHIP   = self._get_pt(lm, P.LEFT_HIP.value, w, h)
        LKNEE  = self._get_pt(lm, P.LEFT_KNEE.value, w, h)
        LANK   = self._get_pt(lm, P.LEFT_ANKLE.value, w, h)
        LSHD   = self._get_pt(lm, P.LEFT_SHOULDER.value, w, h)
        LFOOT  = self._get_pt(lm, P.LEFT_FOOT_INDEX.value, w, h)

        RHIP   = self._get_pt(lm, P.RIGHT_HIP.value, w, h)
        RKNEE  = self._get_pt(lm, P.RIGHT_KNEE.value, w, h)
        RANK   = self._get_pt(lm, P.RIGHT_ANKLE.value, w, h)
        RSHD   = self._get_pt(lm, P.RIGHT_SHOULDER.value, w, h)
        RFOOT  = self._get_pt(lm, P.RIGHT_FOOT_INDEX.value, w, h)

        # 畫腿骨架
        cv2.line(frame, tuple(LHIP),  tuple(LKNEE), (255,255,255), 2)
        cv2.line(frame, tuple(LKNEE), tuple(LANK),  (255,255,255), 2)
        cv2.line(frame, tuple(RHIP),  tuple(RKNEE), (255,255,255), 2)
        cv2.line(frame, tuple(RKNEE), tuple(RANK),  (255,255,255), 2)
        for p,c in ((LHIP,(0,165,255)),(LKNEE,(0,255,0)),(LANK,(255,0,0)),
                    (RHIP,(0,165,255)),(RKNEE,(0,255,0)),(RANK,(255,0,0))):
            cv2.circle(frame, tuple(p), 8, c, -1)

        # 角度（丟進 SMA 再輸出）
        ang = {}
        ang["L_hip"]   = self.hka_sma["L_hip"].push(angle_between(LSHD, LHIP, LKNEE))
        ang["L_knee"]  = self.hka_sma["L_knee"].push(angle_between(LHIP, LKNEE, LANK))
        ang["L_ankle"] = self.hka_sma["L_ankle"].push(angle_between(LKNEE, LANK, LFOOT))
        ang["R_hip"]   = self.hka_sma["R_hip"].push(angle_between(RSHD, RHIP, RKNEE))
        ang["R_knee"]  = self.hka_sma["R_knee"].push(angle_between(RHIP, RKNEE, RANK))
        ang["R_ankle"] = self.hka_sma["R_ankle"].push(angle_between(RKNEE, RANK, RFOOT))
        return ang

    def _overlay_angles_top_left(self, frame, angles, font_scale, line_h, pad):
        lines = [
            f"L-Hip:   {angles['L_hip']:.2f}",
            f"L-Knee:  {angles['L_knee']:.2f}",
            f"L-Ankle: {angles['L_ankle']:.2f}",
            f"R-Hip:   {angles['R_hip']:.2f}",
            f"R-Knee:  {angles['R_knee']:.2f}",
            f"R-Ankle: {angles['R_ankle']:.2f}",
        ]
        text_sizes = [cv2.getTextSize(t, FONT, font_scale, HUD_TEXT_THICK)[0] for t in lines]
        max_w = max(w for w, h in text_sizes)
        box_w = max_w + pad*2;  box_h = pad*2 + line_h*len(lines)
        x0, y0 = 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+box_w, y0+box_h), PANEL_BG_COLOR, -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1-PANEL_ALPHA, 0, frame)
        y = y0 + pad + line_h - int(line_h*0.3)
        for ln in lines:
            cv2.putText(frame, ln, (x0+pad, y), FONT, font_scale, HUD_TEXT_COLOR, HUD_TEXT_THICK, cv2.LINE_AA)
            y += line_h

    def _finish(self):
        self.timer.stop(); self.cap.release()
        if self.writer: self.writer.release()
        self.pose.close()
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！"); QtWidgets.qApp.quit()

# ========================= In-Out-Toeing Viewer（含骨架 + 面板） =========================
class InOutViewer(QtWidgets.QMainWindow):
    def __init__(self, video_path: str, out_path: Optional[str], text_px: int):
        super().__init__()
        self.text_px = max(8, int(text_px))
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): raise RuntimeError(f"無法開啟影片：{video_path}")
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resize(self.width, self.height)
        self.writer = None
        if out_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.width, self.height))
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter); self.setCentralWidget(self.label)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2,
                                           min_detection_confidence=0.6, min_tracking_confidence=0.6,
                                           smooth_landmarks=True)
        self.prev = None  # In/Out 的 EMA 與 body-forward 狀態
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._next_frame)
        self.timer.start(int(1000 / self.fps))

    def _next_frame(self):
        ok, frame = self.cap.read()
        if not ok: self._finish(); return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            analysis, self.prev = analyze_inout_frame(lm, prev=self.prev)

            # ① 畫骨架
            du = mp.solutions.drawing_utils
            du.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                du.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),  # 點：白
                du.DrawingSpec(color=(180, 0, 255), thickness=2),                    # 線：紫
            )

            # ② 足向量箭頭 + 身體前向箭頭
            h, w = frame.shape[:2]
            for side, (heel_idx, toe_idx, color) in (
                ("left", (29,31, (0,255,0))),
                ("right",(30,32, (0,0,255))),
            ):
                heel = lm[heel_idx]; toe = lm[toe_idx]
                heel_xy = np.array([heel.x*w, heel.y*h]);  toe_xy = np.array([toe.x*w, toe.y*h])
                v = toe_xy - heel_xy;  v = v / (np.linalg.norm(v)+1e-6)
                draw_arrow(frame, heel_xy, v, color=color, scale=100, thickness=2)
            hc = analysis["hip_center_xy"]; bf = analysis["bf_vec"]
            draw_body_forward(frame, (hc[0]*w, hc[1]*h), bf, scale=100, color=(255,0,255))

            # ③ 左上面板
            fs, line_h, pad = fixed_font_metrics(self.text_px)
            lines = build_inout_lines(analysis["left_leg"], analysis["right_leg"])
            text_sizes = [cv2.getTextSize(t, FONT, fs, HUD_TEXT_THICK)[0] for t in lines]
            max_w = max(wd for wd, hh in text_sizes)
            box_w = max_w + pad*2;  box_h = pad*2 + line_h*len(lines)
            x0, y0 = 10, 10
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x0+box_w, y0+box_h), PANEL_BG_COLOR, -1)
            cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1-PANEL_ALPHA, 0, frame)
            y = y0 + pad + line_h - int(line_h*0.3)
            for ln in lines:
                cv2.putText(frame, ln, (x0+pad, y), FONT, fs, HUD_TEXT_COLOR, HUD_TEXT_THICK, cv2.LINE_AA)
                y += line_h

        if self.writer: self.writer.write(frame)
        qimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_BGR888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _finish(self):
        self.timer.stop(); self.cap.release()
        if self.writer: self.writer.release()
        self.pose.close()
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！"); QtWidgets.qApp.quit()

# ========================= 主入口 =========================
def main():
    ap = argparse.ArgumentParser(description="Auto Gait Router: tiptoe / hka / inout")
    ap.add_argument("--video", help="影片路徑；空白會直接開檔案選擇視窗")
    ap.add_argument("--out",   default="", help="輸出影片；留空=自動 *_annotated.mp4；'none' 不存檔")
    ap.add_argument("--force", choices=["auto","tiptoe","hka","inout","longshort","lld","intoe","outtoe"],
                    default="auto", help="強制模式（auto=自動；inout/intoe/outtoe 都跑內/外八；longshort/lld=HKA）")
    ap.add_argument("--text-px", type=int, default=DEFAULT_TEXT_PX, help="固定字高（像素），預設 10")
    args = ap.parse_args()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # 影片路徑：未提供就開檔選擇器
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
        root, _ = os.path.splitext(video_path); out_path = root + "_annotated.mp4"
    if out_path.lower() in ("none","null","no","-"): out_path = None

    # 模式（預設 auto；仍保留 --force 以備需要）
    mode = args.force
    if mode == "auto":
        mode = classify_video(video_path)  # 'tiptoe' / 'inout' / 'hka'
    if mode in ("longshort","lld"): mode = "hka"
    if mode in ("intoe","outtoe"):  mode = "inout"

    text_px = max(8, int(args.text_px))
    if mode == "tiptoe":
        viewer = TiptoeViewer(video_path, out_path, text_px)
        viewer.setWindowTitle("Gait Analyzer – Tiptoe")
    elif mode == "inout":
        viewer = InOutViewer(video_path, out_path, text_px)
        viewer.setWindowTitle("Gait Analyzer – In/Out-Toeing")
    else:
        viewer = HKAViewer(video_path, out_path, text_px)
        viewer.setWindowTitle("Gait Analyzer – HKA (Long/Short)")
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

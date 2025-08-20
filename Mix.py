#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto Gait Router：Tiptoe / HKA（長短腳角度）/ In-Out-Toeing（內外八）/ XO（X/O 型腿）
----------------------------------------------------------------------------------
• 前置分類：讀取前 N 幀（含暖機 BURN_IN），只分流到「一種」視圖（不混顯）。
    1) Tiptoe（踮腳）：右上角 HUD。
    2) HKA（髖膝踝六行）：左上角面板。
    3) In-Out-Toeing（內/外八 + 髖內旋）：左上角面板。
    4) XO（X/O 型腿：膝距/踝距）：左上角面板；✱ 保留 xo.py 的文字顏色，僅統一字體/透明白底。

• 疊字與樣式（四模式一致）
    - 全部使用 OpenCV 疊字；字級為「固定像素」(--text-px)，不隨影像大小改變。
    - HKA / InOut / XO 面板固定左上；Tiptoe HUD 固定右上。
    - XO 文字顏色沿用：異常=紅、正常=綠；其餘模式用 HUD_TEXT_COLOR。

    # 自動分類四選一
    python Mix.py

    # 強制某模式
    python Mix.py --force tiptoe
    python Mix.py --force inout
    python Mix.py --force hka
    python Mix.py --force xo

• 輸入：未指定 --video 時開檔選單。
• 輸出：未指定 --out → <來源>_annotated.mp4；--out none 不存檔。
"""

from __future__ import annotations
import sys, os, cv2, argparse
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Optional, Any, Dict, Tuple
from math import acos, degrees
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

# === Mediapipe 繪圖工具（In/Out 會畫骨架） ===
mp_drawing = mp.solutions.drawing_utils
mp_pose     = mp.solutions.pose

# ========================= 統一外觀設定 =========================
FONT            = cv2.FONT_HERSHEY_COMPLEX
HUD_TEXT_COLOR  = (255, 0, 0)     # BGR（藍色系字體）
HUD_TEXT_THICK  = 1
PANEL_BG_COLOR  = (255, 255, 255)
PANEL_ALPHA     = 0.75            # 四模式一致
PANEL_W_RATIO   = 0.50            # 只用在 Tiptoe 右上 HUD
SEG_COLOR       = (180, 180, 180)
DOT_COLOR       = (0, 200, 0)
DOT_RADIUS      = 4

# ======= 字級：固定像素（不依影像大小伸縮） =======
DEFAULT_TEXT_PX = 10  # 可用 --text-px 變更
def fixed_font_metrics(target_px: int, thickness: int = HUD_TEXT_THICK, sample_text: str = "Hg"):
    target_px = max(8, int(target_px))
    ((_, base_h), _) = cv2.getTextSize(sample_text, FONT, 1.0, thickness)
    font_scale = float(target_px) / max(1, base_h)
    line_h = int(target_px + max(4, target_px * 0.2))
    pad    = int(max(8, target_px * 0.6))
    return font_scale, line_h, pad

# ========================= 平滑與門檻（穩定版，固定數值） =========================
# Tiptoe / InOut 都會使用到
PF_THRESHOLD_DEG = 10.0         # Tiptoe 踝角門檻（較保守）
SMA_WIN          = 11           # 踝角 SMA 視窗
EMA2D_ALPHA      = 0.18         # 2D 關鍵點 EMA（穩定）
EMA3D_ALPHA      = 0.18         # 3D/World 點 EMA（穩定）
ANGLE_DEF        = 'KAT'        # 'KAT' 或 'HAT'

# ========================= 自動分流參數（穩定版） =========================
CLS_MAX_FRAMES  = 600           # 觀察上限
CLS_MIN_VALID   = 180           # 至少 ~6 秒有效步態
TIPTOE_RATIO_TH = 0.25          # Tiptoe 佔比門檻
INOUT_RATIO_TH  = 0.55          # In/Out 佔比門檻（偏嚴）
XO_RATIO_TH     = 0.35          # XO 佔比門檻（中等）
BURN_IN         = 45            # 暖機幀：前 45 幀不計入比例

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

# ========================= In/Out-Toeing（內外八） =========================
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
    if th is None:
        th = {"high": 8.0, "low": 3.0, "ema": 0.18, "hip_rot": 15.0}  # 穩定一點的 EMA
    if prev is None:
        prev = {"left_leg":"Neutral","right_leg":"Neutral",
                "ema_angle":{"left":None,"right":None},
                "bf_vec":None,"hip_trace":deque(maxlen=15)}
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
    for side, idxs in (("left",(29,31)), ("right",(30,32))):
        heel = lm[idxs[0]]; toe = lm[idxs[1]]
        foot_vec_raw = np.array([toe.x-heel.x, toe.y-heel.y, toe.z-heel.z], float)
        foot_proj = _normalize(np.array([foot_vec_raw[0], 0.0, foot_vec_raw[2]]))
        raw = prev["ema_angle"][side] or 0.0 if (np.linalg.norm(bf)<1e-5 or np.linalg.norm(foot_proj)<1e-5) \
              else _ang(foot_proj, bf)
        alpha = th["ema"]; prev_val = prev["ema_angle"][side]
        ema = raw if prev_val is None else alpha*raw + (1-alpha)*prev_val
        prev["ema_angle"][side] = ema
        if prev.get(f"{side}_leg") == "In-Toed":
            label = "In-Toed" if ema < th["low"] else "Neutral"
        elif prev.get(f"{side}_leg") == "Out-Toed":
            label = "Out-Toed" if ema > th["high"] else "Neutral"
        else:
            label = "In-Toed" if ema < th["low"] else ("Out-Toed" if ema > th["high"] else "Neutral")
        prev[f"{side}_leg"] = label
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

# ========================= XO（X/O 型腿） =========================
O_LEG_KNEE_THRESHOLD_CM = 6.0
X_LEG_ANKLE_THRESHOLD_CM = 8.0
ASSUMED_HEIGHT_CM = 150.0

def _dist_px(p1, p2, W, H):
    x1, y1 = p1[0]*W, p1[1]*H
    x2, y2 = p2[0]*W, p2[1]*H
    return float(np.hypot(x2-x1, y2-y1))

def _leg_points_from_lm(lm2d):
    try:
        return {
            'left_hip':   (lm2d[23].x, lm2d[23].y),
            'right_hip':  (lm2d[24].x, lm2d[24].y),
            'left_knee':  (lm2d[25].x, lm2d[25].y),
            'right_knee': (lm2d[26].x, lm2d[26].y),
            'left_ankle': (lm2d[27].x, lm2d[27].y),
            'right_ankle':(lm2d[28].x, lm2d[28].y),
        }
    except Exception:
        return None

def _estimate_height_pixels(points, W, H):
    L = _dist_px(points['left_hip'], points['left_ankle'], W, H)
    R = _dist_px(points['right_hip'], points['right_ankle'], W, H)
    leg_avg = (L+R)/2.0
    return leg_avg * 2.0

def _px_to_cm(dist_px, est_height_px, assumed_cm=ASSUMED_HEIGHT_CM):
    cm_per_px = (assumed_cm / float(est_height_px+1e-6))
    return dist_px * cm_per_px

def analyze_xo(points, W, H):
    knee_px  = _dist_px(points['left_knee'],  points['right_knee'],  W, H)
    ankle_px = _dist_px(points['left_ankle'], points['right_ankle'], W, H)
    est_h_px = _estimate_height_pixels(points, W, H)
    knee_cm  = _px_to_cm(knee_px, est_h_px, ASSUMED_HEIGHT_CM)
    ankle_cm = _px_to_cm(ankle_px, est_h_px, ASSUMED_HEIGHT_CM)
    res = {
        "knee_distance_cm": knee_cm,
        "ankle_distance_cm": ankle_cm,
        "knee_distance_px": knee_px,
        "ankle_distance_px": ankle_px,
        "o_leg": False, "o_severity": "false",
        "x_leg": False, "x_severity": "false",
    }
    if knee_cm > O_LEG_KNEE_THRESHOLD_CM:
        res["o_leg"] = True
        res["o_severity"] = "true" if knee_cm > O_LEG_KNEE_THRESHOLD_CM*1.5 else "mild"
    if ankle_cm > X_LEG_ANKLE_THRESHOLD_CM:
        res["x_leg"] = True
        res["x_severity"] = "true" if ankle_cm > X_LEG_ANKLE_THRESHOLD_CM*1.5 else "mild"
    return res

# ========================= 前置分類（四路自動分流） =========================
def classify_video(video_path: str) -> str:
    """
    回傳 'tiptoe' / 'inout' / 'xo' / 'hka'
    先判 tiptoe，其次 in/out，再來 xo；皆未達門檻則 hka。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose = mp_pose.Pose(
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
    xo_frames     = 0
    inout_prev = None
    i = 0
    while i < CLS_MAX_FRAMES:
        ok, frame = cap.read()
        if not ok: break
        i += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks: continue
        if i <= BURN_IN:
            # 暖機：只餵平滑器，不記比例
            lm2d = res.pose_landmarks.landmark
            _ = ema2d[25].push(np.array([lm2d[25].x*W, lm2d[25].y*H], float))
            continue

        lm2d = res.pose_landmarks.landmark
        world = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None
        def get2d(idx):
            u = lm2d[idx]; return ema2d[idx].push(np.array([u.x*W, u.y*H], float))
        def get3d(idx):
            if world is None:
                x,y = get2d(idx); pt = np.array([x, y, 0.0], float)
            else:
                u = world[idx];   pt = np.array([u.x, u.y, u.z], float)
            return ema3d[idx].push(pt)

        # ---- Tiptoe 指標 ----
        L_KNEE, L_ANK, L_HEEL, L_TOE = 25, 27, 29, 31
        R_KNEE, R_ANK, R_HEEL, R_TOE = 26, 28, 30, 32
        if ANGLE_DEF.upper() == 'KAT':
            L_raw = angle_3pt(get3d(L_KNEE), get3d(L_ANK), get3d(L_TOE))
            R_raw = angle_3pt(get3d(R_KNEE), get3d(R_ANK), get3d(R_TOE))
        else:
            L_raw = angle_3pt(get3d(L_HEEL), get3d(L_ANK), get3d(L_TOE))
            R_raw = angle_3pt(get3d(R_HEEL), get3d(R_ANK), get3d(R_TOE))
        L_pf = smaL.push(L_raw); R_pf = smaR.push(R_raw)
        left_tiptoe  = (L_pf > PF_THRESHOLD_DEG) and (lm2d[L_TOE].y < lm2d[L_HEEL].y)
        right_tiptoe = (R_pf > PF_THRESHOLD_DEG) and (lm2d[R_TOE].y < lm2d[R_HEEL].y)

        # ---- In/Out 指標 ----
        analysis, inout_prev = analyze_inout_frame(lm2d, prev=inout_prev)
        any_inout = (analysis["left_leg"]["status"]  != "Neutral") or \
                    (analysis["right_leg"]["status"] != "Neutral")

        # ---- XO 指標 ----
        pts = _leg_points_from_lm(lm2d)
        any_xo = False
        if pts is not None:
            xo_res = analyze_xo(pts, W, H)
            any_xo = xo_res["o_leg"] or xo_res["x_leg"]

        valid += 1
        if left_tiptoe or right_tiptoe: tiptoe_frames += 1
        if any_inout: inout_frames += 1
        if any_xo:    xo_frames    += 1

    cap.release(); pose.close()
    if valid < max(CLS_MIN_VALID, int(0.3 * (CLS_MAX_FRAMES - BURN_IN))):
        print(f"[Classify] valid={valid} < minimum, fallback=hka")
        return "hka"

    tiptoe_ratio = tiptoe_frames / float(valid)
    inout_ratio  = inout_frames  / float(valid)
    xo_ratio     = xo_frames     / float(valid)
    print(f"[Classify] valid={valid}, tiptoe={tiptoe_frames} ({tiptoe_ratio:.3f}), "
          f"inout={inout_frames} ({inout_ratio:.3f}), xo={xo_frames} ({xo_ratio:.3f})")

    if tiptoe_ratio >= TIPTOE_RATIO_TH: return "tiptoe"
    if inout_ratio  >= INOUT_RATIO_TH:  return "inout"
    if xo_ratio     >= XO_RATIO_TH:     return "xo"
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
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
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
            # 畫最小骨架（兩側）
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

# ========================= HKA Viewer（左上面板） =========================
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
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                                 min_detection_confidence=0.6, min_tracking_confidence=0.6,
                                 smooth_landmarks=True)
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

    def _compute_hka_angles(self, frame, landmarks):
        h, w = frame.shape[:2]
        pts = [(lm.x * w, lm.y * h, lm.visibility) for lm in landmarks.landmark]
        pose = mp_pose.PoseLandmark
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
            cv2.line(frame, hip,  knee, (255,255,255), 2);  cv2.line(frame, knee, ankle, (255,255,255), 2)
            cv2.circle(frame, hip,8,(0,165,255),-1); cv2.circle(frame, knee,8,(0,255,0),-1); cv2.circle(frame, ankle,8,(255,0,0),-1)
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

# ========================= In-Out-Toeing Viewer（左上面板） =========================
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
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                                 min_detection_confidence=0.6, min_tracking_confidence=0.6,
                                 smooth_landmarks=True)
        self.prev = None
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._next_frame)
        self.timer.start(int(1000 / self.fps))

    def _next_frame(self):
        ok, frame = self.cap.read()
        if not ok: self._finish(); return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # === 畫出全身骨架關鍵點與連線（使用預設樣式，避免版本相容性問題） ===
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            analysis, self.prev = analyze_inout_frame(lm, prev=self.prev)
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

            # 左上面板（與其它模式同字級/透明度）
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

# ========================= XO Viewer（左上面板；保留文字顏色） =========================
class XOViewer(QtWidgets.QMainWindow):
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
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                                 min_detection_confidence=0.6, min_tracking_confidence=0.6,
                                 smooth_landmarks=True)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._next_frame)
        self.timer.start(int(1000 / self.fps))

    def _next_frame(self):
        ok, frame = self.cap.read()
        if not ok: self._finish(); return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            lm2d = res.pose_landmarks.landmark
            pts = _leg_points_from_lm(lm2d)
            if pts is not None:
                h, w = frame.shape[:2]
                def P(key): return (int(pts[key][0]*w), int(pts[key][1]*h))
                # 骨架（下肢）
                cv2.line(frame, P('left_hip'),  P('left_knee'),  (255,0,0), 2)
                cv2.line(frame, P('left_knee'), P('left_ankle'), (255,0,0), 2)
                cv2.line(frame, P('right_hip'), P('right_knee'), (0,0,255), 2)
                cv2.line(frame, P('right_knee'),P('right_ankle'),(0,0,255), 2)
                # 距離線
                cv2.line(frame, P('left_knee'),  P('right_knee'),  (0,255,255), 3)
                cv2.line(frame, P('left_ankle'), P('right_ankle'), (255,255,0), 3)
                # 點
                for key, color in {'left_hip':(255,0,0),'right_hip':(255,0,0),
                                   'left_knee':(0,255,0),'right_knee':(0,255,0),
                                   'left_ankle':(0,0,255),'right_ankle':(0,0,255)}.items():
                    cv2.circle(frame, P(key), 6, color, -1)

                res_xo = analyze_xo(pts, w, h)
                if res_xo["o_leg"]:
                    o_text  = f"O-shaped legs: {res_xo['o_severity']} ({res_xo['knee_distance_cm']:.1f}cm)"
                    o_color = (0,0,255)   # 紅：異常
                else:
                    o_text  = f"O-shaped legs: normal ({res_xo['knee_distance_cm']:.1f}cm)"
                    o_color = (0,255,0)   # 綠：正常

                if res_xo["x_leg"]:
                    x_text  = f"X-shaped legs: {res_xo['x_severity']} ({res_xo['ankle_distance_cm']:.1f}cm)"
                    x_color = (0,0,255)
                else:
                    x_text  = f"X-shaped legs: normal ({res_xo['ankle_distance_cm']:.1f}cm)"
                    x_color = (0,255,0)

                fs, line_h, pad = fixed_font_metrics(self.text_px)
                lines = [o_text, x_text]
                sizes = [cv2.getTextSize(t, FONT, fs, HUD_TEXT_THICK)[0] for t in lines]
                box_w = max(s[0] for s in sizes) + pad*2
                box_h = pad*2 + line_h*len(lines)
                x0, y0 = 10, 10
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x0+box_w, y0+box_h), PANEL_BG_COLOR, -1)
                cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1-PANEL_ALPHA, 0, frame)
                y = y0 + pad + line_h - int(line_h*0.3)
                for t, c in ((o_text, o_color), (x_text, x_color)):
                    cv2.putText(frame, t, (x0+pad, y), FONT, fs, c, HUD_TEXT_THICK, cv2.LINE_AA)
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
    ap = argparse.ArgumentParser(description="Auto Gait Router: tiptoe / hka / inout / xo")
    ap.add_argument("--video", help="影片路徑；空白會直接開檔案選擇視窗")
    ap.add_argument("--out",   default="", help="輸出影片；留空=自動 *_annotated.mp4；'none' 不存檔")
    ap.add_argument("--force", choices=["auto","tiptoe","hka","inout","xo",
                                        "longshort","lld","intoe","outtoe","xleg","oleg"],
                    default="auto",
                    help="強制模式（auto=自動；inout/intoe/outtoe 都跑內/外八；longshort/lld=HKA；xleg/oleg=XO）")
    ap.add_argument("--text-px", type=int, default=DEFAULT_TEXT_PX, help="固定字高（像素），預設 10")
    args = ap.parse_args()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # 影片：未提供就開檔案對話框
    if not args.video:
        fname, _ = QFileDialog.getOpenFileName(None, "選擇影片", "", "Video Files (*.mp4 *.avi *.mov)")
        if not fname:
            print("未選擇影片，程式結束。"); return
        video_path = fname
    else:
        video_path = args.video.strip('"')

    # 輸出
    out_path = args.out.strip('"')
    if not out_path:
        root, _ = os.path.splitext(video_path); out_path = root + "_annotated.mp4"
    if out_path.lower() in ("none","null","no","-"): out_path = None

    # 模式決定
    mode = args.force
    if mode == "auto":
        mode = classify_video(video_path)  # 'tiptoe' / 'inout' / 'xo' / 'hka'
    if mode in ("longshort","lld"): mode = "hka"
    if mode in ("intoe","outtoe"):  mode = "inout"
    if mode in ("xleg","oleg"):     mode = "xo"

    text_px = max(8, int(args.text_px))
    if mode == "tiptoe":
        viewer = TiptoeViewer(video_path, out_path, text_px)
        viewer.setWindowTitle("Gait Analyzer – Tiptoe")
    elif mode == "inout":
        viewer = InOutViewer(video_path, out_path, text_px)
        viewer.setWindowTitle("Gait Analyzer – In/Out-Toeing")
    elif mode == "xo":
        viewer = XOViewer(video_path, out_path, text_px)
        viewer.setWindowTitle("Gait Analyzer – XO (X/O-shaped legs)")
    else:
        viewer = HKAViewer(video_path, out_path, text_px)
        viewer.setWindowTitle("Gait Analyzer – HKA (Long/Short)")
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

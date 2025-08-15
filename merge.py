#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Gait & Leg Analyzer (with In/Out-Toeing)
-------------------------------------------------
給一支影片，可以選擇或自動決定要跑：
1) tiptoe  : 踮腳尖（TTW）偵測與統計（輸出 *_ttw.mp4、*.ttw.json）
2) xo      : X/O 型腿（醫學距離閾值）評估與可視化（輸出 *_xo.mp4、*_xo.json）
3) hka     : H-K-A（Hip/Knee/Ankle）角度疊加與骨架顯示（輸出 *_annotated.mp4）
4) toe     : 內八 / 外八（足部朝向相對身體前進向量），含箭頭與數據（輸出 *_toe.mp4、*_toe.json）

用法：
  # 自動判斷（維持原邏輯：tiptoe / xo / 其他→hka；toe 供手動切換）
  python merge.py --video /path/to/video.mp4

  # 強制跑某模式
  python merge.py --video ... --mode tiptoe
  python merge.py --video ... --mode xo --assumed-height-cm 150
  python merge.py --video ... --mode hka
  python merge.py --video ... --mode toe

  # 也可以不帶 --video，會跳選檔視窗或要求你輸入
  python merge.py

依賴：
  pip install opencv-python mediapipe numpy
"""

import argparse
import json
import math
import sys
from collections import deque, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, Mapping, Sequence

import cv2
import mediapipe as mp
import numpy as np


# -----------------------------
# 共用：MediaPipe Pose 標記索引
# -----------------------------
POSE = mp.solutions.pose.PoseLandmark

L_SHOULDER = POSE.LEFT_SHOULDER.value
R_SHOULDER = POSE.RIGHT_SHOULDER.value
L_HIP = POSE.LEFT_HIP.value
R_HIP = POSE.RIGHT_HIP.value
L_KNEE = POSE.LEFT_KNEE.value
R_KNEE = POSE.RIGHT_KNEE.value
L_ANKLE = POSE.LEFT_ANKLE.value
R_ANKLE = POSE.RIGHT_ANKLE.value
L_HEEL = POSE.LEFT_HEEL.value
R_HEEL = POSE.RIGHT_HEEL.value
L_FI = POSE.LEFT_FOOT_INDEX.value
R_FI = POSE.RIGHT_FOOT_INDEX.value

FONT = cv2.FONT_HERSHEY_SIMPLEX


# -----------------------------
# 共用：幾何與工具
# -----------------------------
def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """回傳 ∠ABC（度）；a,b,c 為 2D 或 3D numpy 向量"""
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    cosv = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def path_with_suffix(base: Path, suffix: str) -> Path:
    """在檔名 stem 後加上 suffix（不保留舊副檔名），輸出 .mp4 或 .json 等"""
    return base.with_name(base.stem + suffix)


def safe_fps(cap, default=30.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else default


# ============================================================
# in_out_toeing.py 之分析核心（移植為 CLI 版，不含 PyQt GUI）
# ============================================================
# 參考你原檔的 overlay + analyzer + pose_backend 做法。:contentReference[oaicite:2]{index=2}

_REQUIRED_KEYS = [
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index",
    "left_shoulder","right_shoulder",
]

_MP_LM = mp.solutions.pose.PoseLandmark
_LM_INDEX = {lm.name.lower(): lm.value for lm in _MP_LM}

def _to_np(lm) -> np.ndarray:
    return (
        np.array([lm.x, lm.y, lm.z], dtype=float)
        if hasattr(lm, "x") else np.asarray(lm, dtype=float)
    )

def _ensure_dict(seq: Sequence[Any]) -> Dict[str, np.ndarray]:
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
    return float(np.degrees(np.arccos(dot)))

def draw_arrow(image, origin_xy, vec_xy, color=(0, 255, 0), scale=100, thickness=2):
    pt1 = tuple(np.round(origin_xy[:2]).astype(int))
    pt2 = tuple(np.round((origin_xy[:2] + vec_xy[:2] * scale)).astype(int))
    cv2.arrowedLine(image, pt1, pt2, color, thickness, tipLength=0.3)

def draw_body_forward(image, center, forward_vec, scale=100, color=(255, 0, 255)):
    origin = np.array([center[0], center[1]])
    direction = np.array([forward_vec[0], forward_vec[2]])  # XZ→畫面XY
    draw_arrow(image, origin, direction, color=color, scale=scale, thickness=3)

def analyze_leg_rotation(
    landmarks: Any,
    previous: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    回傳 per-side 狀態（In-Toed / Out-Toed / Neutral）與角度等數據。
    演算法與你 in_out_toeing.py 一致：以髖中心在 XZ 的位移估算身體前向，
    取足部 heel→toe 的 XZ 投影與前向夾角，並做 EMA 平滑與高/低雙閾判定。:contentReference[oaicite:3]{index=3}
    """
    if isinstance(landmarks, Mapping):
        lm = _normalize_dict_keys(landmarks)
    else:
        lm = _ensure_dict(landmarks)

    th = {"high": 8.0, "low": 3.0, "ema": 0.2, "hip_rot": 15.0}
    if thresholds:
        th.update(thresholds)

    if previous is None:
        previous = {
            "left_leg": "Neutral", "right_leg": "Neutral",
            "ema_angle": {"left": None, "right": None},
            "bf_vec": None, "hip_trace": deque(maxlen=15),
        }

    # 估身體前向（依髖中心的 XZ 連續位移）
    hip_center = (lm["left_hip"] + lm["right_hip"]) / 2.0
    hip_xz = hip_center[[0, 2]]
    previous["hip_trace"].append(hip_xz)

    if len(previous["hip_trace"]) >= 2:
        delta = previous["hip_trace"][-1] - previous["hip_trace"][-2]
        if np.linalg.norm(delta) > 1e-3:
            body_forward = normalize(np.array([delta[0], 0.0, delta[1]]))
            previous["bf_vec"] = body_forward
        else:
            body_forward = previous["bf_vec"] if previous["bf_vec"] is not None else np.array([1.0, 0.0, 0.0])
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

        # EMA 平滑
        alpha = th["ema"]
        ema_prev = previous["ema_angle"][side]
        ema_angle = raw_angle if ema_prev is None else alpha * raw_angle + (1 - alpha) * ema_prev
        previous["ema_angle"][side] = ema_angle

        # 雙閾判定（含狀態回滯）
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
        hip_rot_status = "Internally Rotated" if hip_rot_angle > th["hip_rot"] else "Neutral"

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


class ToeAnalyzer:
    """
    內/外八 CLI 版：逐帧分析 + OpenCV 疊字 + 匯出影片與 JSON。
    以 in_out_toeing.py 的 PoseBackend 與 analyzer 邏輯為基礎。:contentReference[oaicite:4]{index=4}
    """
    def __init__(self, save_video: bool = True, size: Optional[Tuple[int, int]] = None):
        self.save_video = save_video
        self.size = size  # (w, h) 若提供會重採樣

    def run(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")

        fps = safe_fps(cap)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.size:
            w, h = self.size

        writer = None
        if self.save_video:
            out_path = path_with_suffix(video_path, "_toe.mp4")
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=2, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        prev_state: Optional[Dict[str, Any]] = None
        counts_left, counts_right = Counter(), Counter()
        angles_left, angles_right = [], []
        hiprot_left, hiprot_right = [], []
        frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if self.size:
                frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            analysis = {}
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                analysis, prev_state = analyze_leg_rotation(lm, previous=prev_state)

                # 足部箭頭（綠：左；紅：右），身體前向箭頭（紫）
                for side in ("left", "right"):
                    heel = lm[L_HEEL if side == "left" else R_HEEL]
                    toe = lm[L_FI if side == "left" else R_FI]
                    heel_pt = np.array([heel.x * w, heel.y * h])
                    toe_pt  = np.array([toe.x  * w, toe.y  * h])
                    foot_vec = toe_pt - heel_pt
                    norm = np.linalg.norm(foot_vec) + 1e-6
                    foot_vec /= norm
                    draw_arrow(frame, heel_pt, foot_vec,
                               color=(0, 255, 0) if side == "left" else (0, 0, 255),
                               scale=80, thickness=2)

                # 身體前向箭頭
                hip_l, hip_r = lm[L_HIP], lm[R_HIP]
                hip_center = ((hip_l.x + hip_r.x) / 2 * w, (hip_l.y + hip_r.y) / 2 * h)
                if prev_state and prev_state.get("bf_vec") is not None:
                    draw_body_forward(frame, hip_center, prev_state["bf_vec"])

                # 疊字：左右狀態與角度
                l = analysis.get("left_leg", {}); r = analysis.get("right_leg", {})
                box_lines = [
                    f"Left  : {l.get('status','--')}  (foot {l.get('foot_angle_deg',0):.2f}°, hipRot {l.get('hip_rotation_deg',0):.2f}°)",
                    f"Right : {r.get('status','--')}  (foot {r.get('foot_angle_deg',0):.2f}°, hipRot {r.get('hip_rotation_deg',0):.2f}°)",
                ]
                sizes = [cv2.getTextSize(t, FONT, 0.7, 2)[0] for t in box_lines]
                maxw = max(s[0] for s in sizes) + 10
                line_h = max(s[1] for s in sizes) + 8
                cv2.rectangle(frame, (4, 4), (4 + maxw, 4 + line_h * len(box_lines)), (255, 255, 255), -1)
                y = 4 + line_h - 4
                for t in box_lines:
                    cv2.putText(frame, t, (8, y), FONT, 0.7, (0, 0, 200), 2, cv2.LINE_AA)
                    y += line_h

                # 統計
                counts_left[l.get("status", "Neutral")]  += 1
                counts_right[r.get("status", "Neutral")] += 1
                if "foot_angle_deg" in l: angles_left.append(l["foot_angle_deg"])
                if "foot_angle_deg" in r: angles_right.append(r["foot_angle_deg"])
                if "hip_rotation_deg" in l: hiprot_left.append(l["hip_rotation_deg"])
                if "hip_rotation_deg" in r: hiprot_right.append(r["hip_rotation_deg"])

            if writer is not None:
                writer.write(frame)
            frames += 1

        cap.release()
        if writer is not None:
            writer.release()
        pose.close()

        left_total = sum(counts_left.values()) or 1
        right_total = sum(counts_right.values()) or 1

        report = {
            "video": str(video_path),
            "frames": frames,
            "mode": "toe",
            "left": {
                "in_frames": counts_left.get("In-Toed", 0),
                "out_frames": counts_left.get("Out-Toed", 0),
                "neutral_frames": counts_left.get("Neutral", 0),
                "in_ratio": round(counts_left.get("In-Toed", 0) / left_total, 3),
                "out_ratio": round(counts_left.get("Out-Toed", 0) / left_total, 3),
                "avg_foot_angle_deg": round(float(np.mean(angles_left)) if angles_left else 0.0, 2),
                "avg_hip_rotation_deg": round(float(np.mean(hiprot_left)) if hiprot_left else 0.0, 2),
            },
            "right": {
                "in_frames": counts_right.get("In-Toed", 0),
                "out_frames": counts_right.get("Out-Toed", 0),
                "neutral_frames": counts_right.get("Neutral", 0),
                "in_ratio": round(counts_right.get("In-Toed", 0) / right_total, 3),
                "out_ratio": round(counts_right.get("Out-Toed", 0) / right_total, 3),
                "avg_foot_angle_deg": round(float(np.mean(angles_right)) if angles_right else 0.0, 2),
                "avg_hip_rotation_deg": round(float(np.mean(hiprot_right)) if hiprot_right else 0.0, 2),
            },
        }

        json_path = path_with_suffix(video_path, "_toe.json")
        Path(json_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


# ============================================================
# 下面三個模式，沿用你原本 merge.py 的設計（略有整理）
# ============================================================
@dataclass
class ProbeStats:
    tiptoe_ratio: float
    stationary_ratio: float
    shoulder_level_ok_ratio: float
    frames_used: int


class AutoRouter:
    """
    讀取影片前 N 影格，估計三個指標：
      - tiptoe_ratio：有「踮腳姿勢」徵象之影格比例
      - stationary_ratio：移動量低的影格比例（站立較多）
      - shoulder_level_ok_ratio：左右肩水平（正面）之影格比例
    判斷（維持既有行為）：
      1) tiptoe_ratio >= 0.30 → tiptoe
      2) 否則 stationary_ratio >= 0.60 且 shoulder_level_ok_ratio >= 0.6 → xo
      3) 其餘 → hka
    （toe 模式保留手動切換，不更動原自動路由。）:contentReference[oaicite:5]{index=5}
    """
    PF_THRESHOLD_DEG = 8.0
    PROBE_MAX_FRAMES = 360
    MOVE_THRESH = 0.003
    SHOULDER_LEVEL_DELTA = 0.03

    def probe(self, video_path: Path) -> ProbeStats:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")

        fps = safe_fps(cap)
        max_frames = min(self.PROBE_MAX_FRAMES, int(fps * 12))

        pose = mp.solutions.pose.Pose(
            model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        tiptoe_flags, stationary_flags, shoulder_ok_flags = [], [], []
        prev_lhip = prev_rhip = None
        used = 0

        while used < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                tiptoe_flags.append(False)
                stationary_flags.append(False)
                shoulder_ok_flags.append(False)
                used += 1
                continue

            lm = res.pose_landmarks.landmark

            # tiptoe：踝角度 + 腳尖高於腳跟（沿用）
            def ankle_angle_and_toe_higher(side: str) -> Tuple[float, bool]:
                if side == "R":
                    k, a, fi, heel = lm[R_KNEE], lm[R_ANKLE], lm[R_FI], lm[R_HEEL]
                else:
                    k, a, fi, heel = lm[L_KNEE], lm[L_ANKLE], lm[L_FI], lm[L_HEEL]
                ang = angle_between(
                    np.array([k.x, k.y], float),
                    np.array([a.x, a.y], float),
                    np.array([fi.x, fi.y], float),
                )
                return ang, (fi.y < heel.y)

            ang_r, toe_high_r = ankle_angle_and_toe_higher("R")
            ang_l, toe_high_l = ankle_angle_and_toe_higher("L")
            is_tiptoe = ((ang_r > self.PF_THRESHOLD_DEG and toe_high_r) or
                         (ang_l > self.PF_THRESHOLD_DEG and toe_high_l))
            tiptoe_flags.append(bool(is_tiptoe))

            # stationary：兩髖位移
            lhip = np.array([lm[L_HIP].x, lm[L_HIP].y])
            rhip = np.array([lm[R_HIP].x, lm[R_HIP].y])
            if prev_lhip is None:
                stationary_flags.append(False)
            else:
                move = (np.linalg.norm(lhip - prev_lhip) + np.linalg.norm(rhip - prev_rhip)) / 2.0
                stationary_flags.append(move < self.MOVE_THRESH)
            prev_lhip, prev_rhip = lhip, rhip

            # 肩是否水平
            lsh, rsh = lm[L_SHOULDER], lm[R_SHOULDER]
            shoulder_ok_flags.append(abs(lsh.y - rsh.y) < self.SHOULDER_LEVEL_DELTA)

            used += 1

        cap.release()
        pose.close()

        def ratio(lst):
            return float(np.mean(lst)) if lst else 0.0

        return ProbeStats(
            tiptoe_ratio=ratio(tiptoe_flags),
            stationary_ratio=ratio(stationary_flags),
            shoulder_level_ok_ratio=ratio(shoulder_ok_flags),
            frames_used=used,
        )

    def decide(self, stats: ProbeStats) -> str:
        if stats.tiptoe_ratio >= 0.30:
            return "tiptoe"
        if stats.stationary_ratio >= 0.60 and stats.shoulder_level_ok_ratio >= 0.60:
            return "xo"
        return "hka"


class TiptoeDetector:
    PF_THRESHOLD_DEG = 8.0
    EARLY_RISE_PERCENT = 0.30
    FRAME_RATIO_FLAG = 0.50
    WIN_SMOOTH_ANGLE = 7
    WIN_SMOOTH_ANKLE_Y = 7

    def __init__(self, save_video: bool = True):
        self.save_video = save_video

    def run(self, video_path: Path, out_dir: Optional[Path] = None) -> dict:
        out_dir = out_dir or video_path.parent
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")

        fps = safe_fps(cap)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.save_video:
            out_path = path_with_suffix(video_path, "_ttw.mp4")
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        pose = mp.solutions.pose.Pose(
            model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        q_angle = deque(maxlen=self.WIN_SMOOTH_ANGLE)
        q_ankleY = deque(maxlen=self.WIN_SMOOTH_ANKLE_Y)

        flags = []
        heel_rise_events = 0
        gait_start = None
        prev_ankle_y = None
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            is_tiptoe = False
            if res.pose_landmarks:
                lm2d = res.pose_landmarks.landmark
                RK2, RA2, RH2, RT2 = lm2d[R_KNEE], lm2d[R_ANKLE], lm2d[R_HEEL], lm2d[R_FI]

                ang_raw = angle_between(
                    np.array([RK2.x, RK2.y]),
                    np.array([RA2.x, RA2.y]),
                    np.array([RT2.x, RT2.y]),
                )
                q_angle.append(ang_raw)
                ang_sm = sum(q_angle) / len(q_angle)

                q_ankleY.append(RA2.y)
                ankle_sm = sum(q_ankleY) / len(q_ankleY)

                toe_higher = (RT2.y < RH2.y)
                if ang_sm > self.PF_THRESHOLD_DEG and toe_higher:
                    is_tiptoe = True
                    if writer is not None:
                        cv2.putText(frame, "TIPTOE", (50, 60), FONT, 1.5, (0, 0, 255), 3)

                if prev_ankle_y is not None:
                    delta = ankle_sm - prev_ankle_y
                    if delta > 0 and gait_start is None:
                        gait_start = idx
                    if delta < 0 and gait_start is not None:
                        pct = (idx - gait_start) / fps
                        if pct < self.EARLY_RISE_PERCENT:
                            heel_rise_events += 1
                            if writer is not None:
                                cv2.putText(frame, "EARLY HEEL RISE", (50, 120), FONT, 1.0, (255, 0, 0), 2)
                        gait_start = None
                prev_ankle_y = ankle_sm

            flags.append(is_tiptoe)
            if writer is not None:
                writer.write(frame)
            idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        pose.close()

        ratio = float(np.mean(flags)) if flags else 0.0
        report = {
            "video": str(video_path),
            "fps": fps,
            "frames": idx,
            "tiptoe_frames": int(sum(flags)),
            "tiptoe_ratio": round(ratio, 3),
            "overall_tiptoe": ratio >= self.FRAME_RATIO_FLAG,
            "early_heel_rise_events": int(heel_rise_events),
            "mode": "tiptoe",
        }

        json_path = path_with_suffix(video_path, ".ttw.json")
        Path(json_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


class XOAnalyzer:
    O_KNEE_THRESHOLD_CM = 6.0   # 膝間距 > 6cm → O 型
    X_ANKLE_THRESHOLD_CM = 8.0  # 踝間距 > 8cm → X 型

    def __init__(self, assumed_height_cm: float = 150.0, save_video: bool = True):
        self.assumed_height_cm = float(assumed_height_cm)
        self.save_video = save_video

    @staticmethod
    def _px_dist(p1, p2, w, h):
        x1, y1 = p1.x * w, p1.y * h
        x2, y2 = p2.x * w, p2.y * h
        return math.hypot(x2 - x1, y2 - y1)

    def run(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")

        fps = safe_fps(cap)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.save_video:
            out_path = path_with_suffix(video_path, "_xo.mp4")
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        pose = mp.solutions.pose.Pose(
            model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

        knee_cm_list, ankle_cm_list = [], []
        frames = 0

        def draw_overlay(img, lm, knee_cm, ankle_cm, o_leg, x_leg):
            pts = {
                "LHIP": lm[L_HIP], "LK": lm[L_KNEE], "LA": lm[L_ANKLE],
                "RHIP": lm[R_HIP], "RK": lm[R_KNEE], "RA": lm[R_ANKLE],
            }
            colors = {
                "LHIP": (255, 0, 0), "LK": (0, 255, 0), "LA": (0, 0, 255),
                "RHIP": (255, 0, 0), "RK": (0, 255, 0), "RA": (0, 0, 255),
            }
            for k, p in pts.items():
                cv2.circle(img, (int(p.x * w), int(p.y * h)), 6, colors[k], -1)

            def line(pa, pb, color, thick=2):
                cv2.line(img, (int(pa.x * w), int(pa.y * h)),
                              (int(pb.x * w), int(pb.y * h)), color, thick)

            line(pts["LHIP"], pts["LK"], (255, 0, 0))
            line(pts["LK"], pts["LA"], (255, 0, 0))
            line(pts["RHIP"], pts["RK"], (0, 0, 255))
            line(pts["RK"], pts["RA"], (0, 0, 255))

            lk = lm[L_KNEE]; rk = lm[R_KNEE]
            la = lm[L_ANKLE]; ra = lm[R_ANKLE]
            line(lk, rk, (0, 255, 255), 3)   # 膝距
            line(la, ra, (255, 255, 0), 3)   # 踝距

            y = 24
            if o_leg:
                txt = f"O-shaped legs: {('true' if knee_cm > self.O_KNEE_THRESHOLD_CM*1.5 else 'mild')} ({knee_cm:.1f}cm)"
                cv2.putText(img, txt, (10, y), FONT, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(img, f"O-shaped legs: normal ({knee_cm:.1f}cm)", (10, y), FONT, 0.7, (0, 255, 0), 2)
            y += 28
            if x_leg:
                txt = f"X-shaped legs: {('true' if ankle_cm > self.X_ANKLE_THRESHOLD_CM*1.5 else 'mild')} ({ankle_cm:.1f}cm)"
                cv2.putText(img, txt, (10, y), FONT, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(img, f"X-shaped legs: normal ({ankle_cm:.1f}cm)", (10, y), FONT, 0.7, (0, 255, 0), 2)

        def estimated_height_pixels(lm):
            left_leg = self._px_dist(lm[L_HIP], lm[L_ANKLE], w, h)
            right_leg = self._px_dist(lm[R_HIP], lm[R_ANKLE], w, h)
            return (left_leg + right_leg) / 2.0 * 2.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                est_h = estimated_height_pixels(lm)
                if est_h > 0:
                    knee_px = self._px_dist(lm[L_KNEE], lm[R_KNEE], w, h)
                    ankle_px = self._px_dist(lm[L_ANKLE], lm[R_ANKLE], w, h)
                    cm_per_px = self.assumed_height_cm / est_h
                    knee_cm = knee_px * cm_per_px
                    ankle_cm = ankle_px * cm_per_px
                    knee_cm_list.append(knee_cm)
                    ankle_cm_list.append(ankle_cm)

                    o_leg = knee_cm > self.O_KNEE_THRESHOLD_CM
                    x_leg = ankle_cm > self.X_ANKLE_THRESHOLD_CM

                    if writer is not None:
                        draw_overlay(frame, lm, knee_cm, ankle_cm, o_leg, x_leg)

            if writer is not None:
                writer.write(frame)
            frames += 1

        cap.release()
        if writer is not None:
            writer.release()
        pose.close()

        knee_cm_avg = float(np.mean(knee_cm_list)) if knee_cm_list else 0.0
        ankle_cm_avg = float(np.mean(ankle_cm_list)) if ankle_cm_list else 0.0

        report = {
            "video": str(video_path),
            "frames": frames,
            "assumed_height_cm": self.assumed_height_cm,
            "knee_distance_cm_avg": round(knee_cm_avg, 2),
            "ankle_distance_cm_avg": round(ankle_cm_avg, 2),
            "o_leg": knee_cm_avg > self.O_KNEE_THRESHOLD_CM,
            "x_leg": ankle_cm_avg > self.X_ANKLE_THRESHOLD_CM,
            "mode": "xo",
        }
        json_path = path_with_suffix(video_path, "_xo.json")
        Path(json_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


class HKAAnnotator:
    FONT_SCALE = 0.8
    THICK = 2
    BOX_COLOR = (0, 0, 0)
    TEXT_COLOR = (255, 255, 255)

    def __init__(self, save_video: bool = True):
        self.save_video = save_video

    def run(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")

        fps = safe_fps(cap)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.save_video:
            out_path = path_with_suffix(video_path, "_annotated.mp4")
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        pose = mp.solutions.pose.Pose(
            model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )

        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                def pt(idx):
                    return np.array([int(lm[idx].x * w), int(lm[idx].y * h)], dtype=int)

                Lhip, Lknee, Lankle = pt(L_HIP), pt(L_KNEE), pt(L_ANKLE)
                Rhip, Rknee, Rankle = pt(R_HIP), pt(R_KNEE), pt(R_ANKLE)
                Lsh, Lfi = pt(L_SHOULDER), pt(L_FI)
                Rsh, Rfi = pt(R_SHOULDER), pt(R_FI)

                cv2.line(frame, Lhip, Lknee, (255, 255, 255), 2)
                cv2.line(frame, Lknee, Lankle, (255, 255, 255), 2)
                cv2.line(frame, Rhip, Rknee, (255, 255, 255), 2)
                cv2.line(frame, Rknee, Rankle, (255, 255, 255), 2)

                for p, c in [(Lhip, (0,165,255)), (Lknee, (0,255,0)), (Lankle, (255,0,0)),
                             (Rhip, (0,165,255)), (Rknee, (0,255,0)), (Rankle, (255,0,0))]:
                    cv2.circle(frame, p, 8, c, -1)

                angles = {
                    "L_hip": angle_between(Lsh, Lhip, Lknee),
                    "L_knee": angle_between(Lhip, Lknee, Lankle),
                    "L_ankle": angle_between(Lknee, Lankle, Lfi),
                    "R_hip": angle_between(Rsh, Rhip, Rknee),
                    "R_knee": angle_between(Rhip, Rknee, Rankle),
                    "R_ankle": angle_between(Rknee, Rankle, Rfi),
                }

                lines = [
                    f"L-Hip:   {angles['L_hip']:.2f}",
                    f"L-Knee:  {angles['L_knee']:.2f}",
                    f"L-Ankle: {angles['L_ankle']:.2f}",
                    f"R-Hip:   {angles['R_hip']:.2f}",
                    f"R-Knee:  {angles['R_knee']:.2f}",
                    f"R-Ankle: {angles['R_ankle']:.2f}",
                ]
                sizes = [cv2.getTextSize(t, FONT, 0.8, 2)[0] for t in lines]
                max_w = max(w0 for w0, h0 in sizes) + 10
                line_h = max(h0 for w0, h0 in sizes) + 6
                box_h = line_h * len(lines) + 4
                cv2.rectangle(frame, (0, 0), (max_w, box_h), self.BOX_COLOR, -1)
                y = line_h
                for ln in lines:
                    cv2.putText(frame, ln, (5, y), FONT, 0.8, self.TEXT_COLOR, 2, cv2.LINE_AA)
                    y += line_h

            if writer is not None:
                writer.write(frame)
            frames += 1

        cap.release()
        if writer is not None:
            writer.release()
        pose.close()

        return {"video": str(video_path), "frames": frames, "mode": "hka"}


# -----------------------------
# 工具：解析 / 補齊影片路徑
# -----------------------------
def resolve_input_video(cli_value: Optional[str]) -> Path:
    """
    1) 有帶 CLI 參數就用之
    2) 沒帶就試圖跳檔案選擇視窗（有桌面環境時）
    3) 再不行就用命令列 input()
    """
    if cli_value:
        p = Path(cli_value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"找不到影片：{p}")
        return p
    try:
        from tkinter import Tk, filedialog
        Tk().withdraw()
        selected = filedialog.askopenfilename(
            title="選擇影片",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")]
        )
        if selected:
            p = Path(selected).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"找不到影片：{p}")
            return p
    except Exception:
        pass
    try:
        entered = input("請輸入影片路徑：").strip()
    except EOFError:
        entered = ""
    if not entered:
        print("未提供影片路徑，程式結束。")
        sys.exit(2)
    p = Path(entered).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"找不到影片：{p}")
    return p


# -----------------------------
# 主程式
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Unified Gait & Leg Analyzer (with In/Out-Toeing)")
    ap.add_argument("--video", default=None,
                    help="輸入影片路徑（可省略；未提供會跳選擇視窗或要求輸入）")
    ap.add_argument("--mode", choices=["auto", "tiptoe", "xo", "hka", "toe"], default="auto",
                    help="分析模式：auto/tiptoe/xo/hka/toe")
    ap.add_argument("--assumed-height-cm", type=float, default=150.0,
                    help="XO 模式用的假設身高（公分），用於像素→公分換算（預設 150）")
    ap.add_argument("--no-video-out", action="store_true",
                    help="不輸出標註影片（只輸出 JSON/統計）")
    ap.add_argument("--toe-size", default=None,
                    help="toe 模式輸出尺寸，例如 608x1080；留空沿用原影片大小")
    args = ap.parse_args()

    video_path = resolve_input_video(args.video)

    # 自動判斷（保留原行為）:contentReference[oaicite:6]{index=6}
    mode = args.mode
    if mode == "auto":
        router = AutoRouter()
        stats = router.probe(video_path)
        mode = router.decide(stats)
        print("[Auto] probe stats:",
              json.dumps({
                  "tiptoe_ratio": round(stats.tiptoe_ratio, 3),
                  "stationary_ratio": round(stats.stationary_ratio, 3),
                  "shoulder_level_ok_ratio": round(stats.shoulder_level_ok_ratio, 3),
                  "frames_used": stats.frames_used,
                  "decided_mode": mode
              }, ensure_ascii=False))

    # 執行
    if mode == "tiptoe":
        detector = TiptoeDetector(save_video=not args.no_video_out)
        report = detector.run(video_path)
    elif mode == "xo":
        analyzer = XOAnalyzer(assumed_height_cm=args.assumed_height_cm, save_video=not args.no_video_out)
        report = analyzer.run(video_path)
    elif mode == "toe":
        size = None
        if args.toe_size:
            try:
                w, h = args.toe_size.lower().split("x")
                size = (int(w), int(h))
            except Exception:
                raise ValueError("--toe-size 格式需為 WxH，例如 608x1080")
        toe = ToeAnalyzer(save_video=not args.no_video_out, size=size)
        report = toe.run(video_path)
    else:
        annot = HKAAnnotator(save_video=not args.no_video_out)
        report = annot.run(video_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

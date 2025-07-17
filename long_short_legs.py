"""
leg_length_discrepancy_detector.py
使用 MediaPipe Pose + 時序平滑檢測影片中長短腳 (Leg-Length Discrepancy, LLD)

安裝套件:
pip install mediapipe opencv-python numpy scipy pandas tqdm

執行範例:
python leg_length_discrepancy_detector.py --video input.mp4 --smoothing sg \
       --sg-window 15 --sg-poly 2 --length-threshold 0.03 --pelvis-threshold 0.05
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from collections import deque
from tqdm import tqdm
import argparse
import os

# ---------- 參數與常數 ----------
POSE_LANDMARKS = mp.solutions.pose.PoseLandmark
# 取用的關鍵節點索引
L_HIP, R_HIP = POSE_LANDMARKS.LEFT_HIP.value, POSE_LANDMARKS.RIGHT_HIP.value
L_ANK, R_ANK = POSE_LANDMARKS.LEFT_ANKLE.value, POSE_LANDMARKS.RIGHT_ANKLE.value

# ---------- 工具函式 ----------
def euclidean(p1, p2):
    """計算 2D 歐氏距離（輸入為 Mediapipe 規格 normalized coords）。"""
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

def smooth_series(data, method="sg", **kwargs):
    """依指定方法平滑一條序列。"""
    if method == "ema":
        alpha = kwargs.get("alpha", 0.3)
        smoothed = []
        for x in data:
            smoothed.append(x if not smoothed else alpha * x + (1 - alpha) * smoothed[-1])
        return np.array(smoothed)
    # default Savitzky‑Golay
    window = kwargs.get("window", 11)
    poly   = kwargs.get("poly", 2)
    if len(data) < window:
        return np.array(data)          # 序列太短不處理
    return savgol_filter(data, window_length=window, polyorder=poly)

# ---------- 主流程 ----------
def analyse_video(
    video_path:str,
    smoothing:str="sg",
    sg_window:int=15,
    sg_poly:int=2,
    ema_alpha:float=0.3,
    length_th:float=0.03,     # 3 % 相對差異
    pelvis_th:float=0.05):    # 5 % 相對高度差

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    lengths_L, lengths_R, pelvis_delta = [], [], []

    with mp_pose:
        for _ in tqdm(range(frame_count), desc="Processing frames"):
            ok, frame = cap.read()
            if not ok:
                break

            # 轉 RGB 並送入 Mediapipe
            results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = results.pose_landmarks
            if not lm:
                continue

            # 讀取所需座標 (x,y 為 0~1 normalized)
            coords = [(lmk.x, lmk.y, lmk.visibility) for lmk in lm.landmark]
            # 只取可見度 >0.6 節點
            if coords[L_HIP][2] < .6 or coords[R_HIP][2] < .6 or \
               coords[L_ANK][2] < .6 or coords[R_ANK][2] < .6:
                continue

            len_L = euclidean(coords[L_HIP], coords[L_ANK])
            len_R = euclidean(coords[R_HIP], coords[R_ANK])
            lengths_L.append(len_L)
            lengths_R.append(len_R)

            pelvis_delta.append(abs(coords[L_HIP][1] - coords[R_HIP][1]))  # 垂直差距

    cap.release()

    # ---------- 時序平滑 ----------
    if smoothing == "sg":
        lengths_L = smooth_series(lengths_L, "sg", window=sg_window, poly=sg_poly)
        lengths_R = smooth_series(lengths_R, "sg", window=sg_window, poly=sg_poly)
        pelvis_delta = smooth_series(pelvis_delta, "sg", window=sg_window, poly=sg_poly)
    else:
        lengths_L = smooth_series(lengths_L, "ema", alpha=ema_alpha)
        lengths_R = smooth_series(lengths_R, "ema", alpha=ema_alpha)
        pelvis_delta = smooth_series(pelvis_delta, "ema", alpha=ema_alpha)

    # ---------- 統計與判斷 ----------
    # 以身體寬度 (兩髖水平距離) 當比例基準，可降低拍攝距離誤差
    mean_leg = (np.mean(lengths_L) + np.mean(lengths_R)) / 2
    diff_ratio = abs(np.mean(lengths_L) - np.mean(lengths_R)) / mean_leg

    pelvis_ratio = np.mean(pelvis_delta) / mean_leg  # 相對身高單位

    result = {
        "leg_length_mean_L": float(np.mean(lengths_L)),
        "leg_length_mean_R": float(np.mean(lengths_R)),
        "length_diff_ratio": float(diff_ratio),
        "pelvis_tilt_ratio": float(pelvis_ratio),
        "LLD_flag": diff_ratio > length_th or pelvis_ratio > pelvis_th,
    }

    # ---------- 輸出 ----------
    df = pd.DataFrame({
        "LL_length": lengths_L,
        "RL_length": lengths_R,
        "Pelvis_delta": pelvis_delta,
    })
    csv_path = os.path.splitext(video_path)[0] + "_lld.csv"
    df.to_csv(csv_path, index=False)

    return result, csv_path

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--smoothing", choices=["sg", "ema"], default="sg")
    ap.add_argument("--sg-window", type=int, default=15)
    ap.add_argument("--sg-poly", type=int, default=2)
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--length-threshold", type=float, default=0.03)
    ap.add_argument("--pelvis-threshold", type=float, default=0.05)
    args = ap.parse_args()

    res, csv_path = analyse_video(
        args.video,
        args.smoothing,
        args.sg_window,
        args.sg_poly,
        args.ema_alpha,
        args.length_threshold,
        args.pelvis_threshold,
    )

    print("=== 影片分析結果 ===")
    for k, v in res.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"逐幀數據已輸出至 {csv_path}")

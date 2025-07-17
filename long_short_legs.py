"""
leg_length_discrepancy_detector.py
---------------------------------
偵測影片中長短腳 (Leg-Length Discrepancy, LLD)
依賴: mediapipe >=0.10, opencv-python, numpy, scipy, pandas, tqdm
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm
import argparse, os, sys

# ========== 預設參數，可自行修改 ==========
DEFAULT_VIDEO   = "C:\\Users\\User\\Desktop\\0717.mp4"   # 若 CLI 沒傳 --video 就用這支
DEFAULT_SMOOTH  = "sg"       # sg / ema
DEFAULT_LEN_TH  = 0.03       # 腿長差判斷閾值 (比例)
DEFAULT_PEL_TH  = 0.05       # 骨盆傾斜判斷閾值 (比例)
# ==========================================

# ---------- 關鍵點索引 ----------
POSE = mp.solutions.pose.PoseLandmark
L_HIP, R_HIP = POSE.LEFT_HIP.value,  POSE.RIGHT_HIP.value
L_ANK, R_ANK = POSE.LEFT_ANKLE.value, POSE.RIGHT_ANKLE.value

# ---------- 工具函式 ----------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

def smooth_series(arr, method, **kw):
    if method == "ema":
        alpha = kw.get("alpha", 0.3)
        out = []
        for x in arr:
            out.append(x if not out else alpha * x + (1 - alpha) * out[-1])
        return np.array(out)
    # default = Savitzky‑Golay
    win = kw.get("window", 11); poly = kw.get("poly", 2)
    if len(arr) < win: return np.array(arr)
    return savgol_filter(arr, window_length=win, polyorder=poly)

# ---------- 主流程 ----------
def analyse_video(path, smooth, sg_win, sg_poly, ema_alpha,
                  len_th, pel_th):

    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到影片檔：{path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV 無法讀取影片，請確認檔案格式/權限")

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.6)

    len_L, len_R, pel_dy = [], [], []

    with mp_pose:
        for _ in tqdm(range(frame_cnt), desc="解析中"):
            ok, frame = cap.read()
            if not ok: break
            res = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = res.pose_landmarks
            if not lm: continue
            pts = [(p.x, p.y, p.visibility) for p in lm.landmark]
            if min(pts[L_HIP][2], pts[R_HIP][2],
                   pts[L_ANK][2], pts[R_ANK][2]) < 0.6:
                continue
            len_L.append(euclidean(pts[L_HIP], pts[L_ANK]))
            len_R.append(euclidean(pts[R_HIP], pts[R_ANK]))
            pel_dy.append(abs(pts[L_HIP][1] - pts[R_HIP][1]))

    cap.release()
    if not len_L or not len_R:
        raise RuntimeError("影片中未偵測到足夠的可用關鍵點，無法計算 LLD")

    # 平滑
    len_L = smooth_series(len_L, smooth, window=sg_win, poly=sg_poly,
                          alpha=ema_alpha)
    len_R = smooth_series(len_R, smooth, window=sg_win, poly=sg_poly,
                          alpha=ema_alpha)
    pel_dy = smooth_series(pel_dy, smooth, window=sg_win, poly=sg_poly,
                           alpha=ema_alpha)

    mean_leg = (np.mean(len_L) + np.mean(len_R)) / 2
    diff_ratio = abs(np.mean(len_L) - np.mean(len_R)) / mean_leg
    pel_ratio  = np.mean(pel_dy) / mean_leg

    result = dict(
        mean_leg_L = float(np.mean(len_L)),
        mean_leg_R = float(np.mean(len_R)),
        length_diff_ratio = float(diff_ratio),
        pelvis_tilt_ratio = float(pel_ratio),
        LLD_flag = diff_ratio > len_th or pel_ratio > pel_th
    )

    # 逐幀 CSV
    csv_out = os.path.splitext(path)[0] + "_lld.csv"
    pd.DataFrame({"LL":len_L,"RL":len_R,"PelvisDy":pel_dy}).to_csv(csv_out,
                                                                   index=False)
    return result, csv_out

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Leg-Length Discrepancy Detector (MediaPipe Pose)")
    ap.add_argument("--video", default=DEFAULT_VIDEO,
                    help="影片路徑；留空則使用預設檔案")
    ap.add_argument("--smoothing", choices=["sg","ema"],
                    default=DEFAULT_SMOOTH)
    ap.add_argument("--sg-window", type=int, default=15)
    ap.add_argument("--sg-poly",   type=int, default=2)
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--length-threshold", type=float, default=DEFAULT_LEN_TH)
    ap.add_argument("--pelvis-threshold", type=float, default=DEFAULT_PEL_TH)
    args = ap.parse_args()

    try:
        res, csv_path = analyse_video(
            args.video, args.smoothing, args.sg_window,
            args.sg_poly, args.ema_alpha,
            args.length_threshold, args.pelvis_threshold)
    except Exception as e:
        sys.exit(f"[Error] {e}")

    print("\n=== 影片分析結果 ===")
    for k,v in res.items():
        print(f"{k:20}: {v:.4f}" if isinstance(v,float) else f"{k:20}: {v}")
    print(f"逐幀數據已輸出：{csv_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
以 MediaPipe Pose 偵測影片中的長短腳 (Leg‐Length Discrepancy, LLD)。
偵測到時在畫面標註並輸出 *_annotated.mp4；同時輸出逐幀 CSV。
依賴：mediapipe >= 0.10, opencv‐python, numpy, scipy, pandas, tqdm

⚠️ 變更：
── 2025‑07‑18 ─────────────────────────────────────────────────────────────
1. 偵測到 LLD 時的視覺化方式，從「紅色粗線」改為
   「白色骨架線 + 彩色關鍵點」
     • 髖關節 → 橙色 (BGR = 0,165,255)
     • 膝關節 → 綠色 (BGR = 0,255,0)
     • 踝關節 → 藍色 (BGR = 255,0,0)
2. 新增 KNEE landmark 常數。
"""

import cv2, mediapipe as mp, numpy as np, pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm
import argparse, os, sys, textwrap

# ---------- 預設 ----------
DEFAULT_VIDEO   = "C:\\Users\\User\\Desktop\\Long and short legs111111.mp4"
DEFAULT_SMOOTH  = "sg"            # sg / ema
DEFAULT_LEN_TH  = 0.03            # 腿長差判斷閾值 (比例)
DEFAULT_PEL_TH  = 0.05            # 骨盆傾斜判斷閾值 (比例)
# -------------------------

POSE = mp.solutions.pose.PoseLandmark
L_HIP,  R_HIP  = POSE.LEFT_HIP.value,    POSE.RIGHT_HIP.value
L_KNEE, R_KNEE = POSE.LEFT_KNEE.value,   POSE.RIGHT_KNEE.value  # ★ 新增
L_ANK,  R_ANK  = POSE.LEFT_ANKLE.value,  POSE.RIGHT_ANKLE.value

# ---------- 工具 ----------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))


def smooth_series(arr, method, **kw):
    """回傳 NumPy array；不足長度時原樣返回"""
    if method == "ema":
        alpha = kw.get("alpha", 0.3)
        out = []
        for x in arr:
            out.append(x if not out else alpha * x + (1 - alpha) * out[-1])
        return np.asarray(out, dtype=float)

    win  = kw.get("window", 11)
    poly = kw.get("poly", 2)
    if len(arr) < win:
        return np.asarray(arr, dtype=float)
    return savgol_filter(arr, window_length=win, polyorder=poly)


def put_multiline(img, text, origin, *, font_scale=0.75,
                  thickness=2, color=(0, 0, 255)):
    x, y0 = origin
    for i, line in enumerate(textwrap.dedent(text).splitlines()):
        y = y0 + i * int(30 * font_scale)
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


# ---------- 主流程 ----------

def analyse_and_annotate(
    path: str,
    smooth: str,
    sg_win: int,
    sg_poly: int,
    ema_alpha: float,
    len_th: float,
    pel_th: float,
    out_path: str,
):

    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到影片檔：{path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV 無法讀取影片")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, vw, fps, (W, H))

    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    len_L, len_R, pel_dy, per_frame_flag = [], [], [], []

    with mp_pose:
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="處理影片中")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = res.pose_landmarks
            flag = False

            if lm:
                pts = [(p.x, p.y, p.visibility) for p in lm.landmark]
                if min(
                    pts[L_HIP][2],
                    pts[R_HIP][2],
                    pts[L_KNEE][2],
                    pts[R_KNEE][2],
                    pts[L_ANK][2],
                    pts[R_ANK][2],
                ) >= 0.6:

                    def px(pt):  # 轉成像素座標
                        return np.array([pt[0] * W, pt[1] * H], dtype=int)

                    # --- 量測長度與骨盆差 ---
                    LL = euclidean(pts[L_HIP], pts[L_ANK])
                    RL = euclidean(pts[R_HIP], pts[R_ANK])
                    pdy = abs(pts[L_HIP][1] - pts[R_HIP][1])

                    len_L.append(LL)
                    len_R.append(RL)
                    pel_dy.append(pdy)

                    mean_leg = (LL + RL) / 2.0
                    diff_ratio = abs(LL - RL) / mean_leg
                    pel_ratio = pdy / mean_leg
                    flag = diff_ratio > len_th or pel_ratio > pel_th
                    per_frame_flag.append(flag)

                    # ---------- 視覺化 ----------
                    if flag:
                        short_is_left = LL < RL
                        hip = px(pts[L_HIP] if short_is_left else pts[R_HIP])
                        knee = px(pts[L_KNEE] if short_is_left else pts[R_KNEE])
                        ankle = px(pts[L_ANK] if short_is_left else pts[R_ANK])

                        # 白色骨架線
                        cv2.line(frame, tuple(hip), tuple(knee), (255, 255, 255), thickness=2)
                        cv2.line(frame, tuple(knee), tuple(ankle), (255, 255, 255), thickness=2)

                        # 彩色關鍵點
                        cv2.circle(frame, tuple(hip), 8, (0, 165, 255), -1)   # orange – HIP
                        cv2.circle(frame, tuple(knee), 8, (0, 255, 0), -1)    # green  – KNEE
                        cv2.circle(frame, tuple(ankle), 8, (255, 0, 0), -1)   # blue   – ANKLE

                        leg_txt = "L" if short_is_left else "R"
                        put_multiline(
                            frame, f"LLD DETECTED\n(Shorter leg: {leg_txt})", (20, 40)
                        )
                else:
                    per_frame_flag.append(False)
                    put_multiline(
                        frame,
                        "POSE NOT DETECTED",
                        (20, 40),
                        font_scale=0.8,
                        color=(0, 255, 255),
                    )
            else:
                per_frame_flag.append(False)
                put_multiline(
                    frame,
                    "POSE NOT DETECTED",
                    (20, 40),
                    font_scale=0.8,
                    color=(0, 255, 255),
                )

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()

    # ---------- 平滑 ----------
    if len(len_L) and len(len_R):
        len_L_s = smooth_series(len_L, smooth, window=sg_win, poly=sg_poly, alpha=ema_alpha)
        len_R_s = smooth_series(len_R, smooth, window=sg_win, poly=sg_poly, alpha=ema_alpha)
        pel_dy_s = smooth_series(pel_dy, smooth, window=sg_win, poly=sg_poly, alpha=ema_alpha)
        has_data = len(len_L_s) > 0
    else:
        # 沒偵測到姿勢
        len_L_s = len_R_s = pel_dy_s = np.asarray([])
        has_data = False

    if has_data:
        mean_leg = (np.mean(len_L_s) + np.mean(len_R_s)) / 2.0
        diff_ratio = abs(np.mean(len_L_s) - np.mean(len_R_s)) / mean_leg
        pel_ratio = np.mean(pel_dy_s) / mean_leg
        overall_flag = diff_ratio > len_th or pel_ratio > pel_th
    else:
        diff_ratio = pel_ratio = 0.0
        overall_flag = False

    # ---------- CSV ----------
    csv_path = None
    if has_data:
        csv_path = os.path.splitext(path)[0] + "_lld.csv"
        df = pd.DataFrame(
            {
                "LL": len_L_s,
                "RL": len_R_s,
                "PelvisDy": pel_dy_s,
                "LLD_Frame_Flag": per_frame_flag[: len(len_L_s)],
            }
        )
        df.to_csv(csv_path, index=False)

    return dict(
        mean_leg_L=float(np.mean(len_L_s)) if has_data else None,
        mean_leg_R=float(np.mean(len_R_s)) if has_data else None,
        length_diff_ratio=float(diff_ratio) if has_data else None,
        pelvis_tilt_ratio=float(pel_ratio) if has_data else None,
        LLD_flag=overall_flag,
        annotated_video=out_path,
        csv=csv_path,
    )


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Leg-Length Discrepancy Annotator (MediaPipe Pose)")
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="影片路徑；留空則使用預設檔案")
    ap.add_argument("--output", default=None, help="標註後影片輸出路徑 (預設 *_annotated.mp4)")
    ap.add_argument("--smoothing", choices=["sg", "ema"], default=DEFAULT_SMOOTH)
    ap.add_argument("--sg-window", type=int, default=15)
    ap.add_argument("--sg-poly", type=int, default=2)
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--length-threshold", type=float, default=DEFAULT_LEN_TH)
    ap.add_argument("--pelvis-threshold", type=float, default=DEFAULT_PEL_TH)
    args = ap.parse_args()

    video_path = args.video
    out_path = args.output if args.output else os.path.splitext(video_path)[0] + "_annotated.mp4"

    try:
        res = analyse_and_annotate(
            video_path,
            args.smoothing,
            args.sg_window,
            args.sg_poly,
            args.ema_alpha,
            args.length_threshold,
            args.pelvis_threshold,
            out_path,
        )
    except Exception as e:
        sys.exit(f"[Error] {e}")

    print("\n=== 影片分析結果 ===")
    for k, v in res.items():
        print(f"{k:20}: {v:.4f}" if isinstance(v, float) else f"{k:20}: {v}")


if __name__ == "__main__":
    main()

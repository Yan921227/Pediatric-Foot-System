#!/usr/bin/env python
"""
Leg-Length Discrepancy (LLD) detector — visual + bug‑fixed 版
對應表１診斷標準：
    - 腿長差  > 2 %
    - 骨盆傾斜 > 5°
    - 步長差  > 5 cm（以比例 ~5.5 %）
★ 重新加入：
    - 白色骨架線（髖→膝→踝，短腿側）
    - 關鍵點：髖=橙、膝=綠、踝=藍
    - 左上角「LLD DETECTED / Shorter leg: L|R」文字
"""

import cv2, mediapipe as mp, numpy as np, pandas as pd, math, os, sys, argparse, textwrap
from tqdm import tqdm

# ── 閾值（表格對應） ────────────────────────────────
DEFAULT_VIDEO          = "C:\\Users\\User\\Desktop\\Long and short legs111111.mp4"
LEN_DIFF_RATIO_TH      = 0.02          # 腿長差 2 %
PEL_ANGLE_TH_DEG       = 5.0           # 骨盆傾斜 5°
STEP_DIFF_RATIO_TH     = 0.055         # 步長差 ≈ 5 cm / 90 cm
TEXT_SCALE_ALERT       = 1.6           # ★ LLD 警示文字放大倍率
# ────────────────────────────────────────────────

POSE = mp.solutions.pose.PoseLandmark
L_HIP,  R_HIP  = POSE.LEFT_HIP.value,   POSE.RIGHT_HIP.value
L_KNEE, R_KNEE = POSE.LEFT_KNEE.value,  POSE.RIGHT_KNEE.value
L_ANK,  R_ANK  = POSE.LEFT_ANKLE.value, POSE.RIGHT_ANKLE.value

# ── 工具 ──────────────────────────────────────────
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

def put_multiline(img, txt, org, *, fs=0.8, col=(0,0,255)):
    x, y0 = org
    for i, ln in enumerate(textwrap.dedent(txt).splitlines()):
        cv2.putText(img, ln, (x, y0 + i*int(28*fs)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2, cv2.LINE_AA)
# ────────────────────────────────────────────────

def analyse_and_annotate(path: str, out_path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV 無法開啟影片")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.6)

    # 序列（保證一一對應）
    len_L=[]; len_R=[]; pel_angle=[]
    step_proj_L=[]; step_proj_R=[]; frame_flag=[]

    with mp_pose:
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Analysing")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 預先填 NaN 佔位
            len_L.append(np.nan); len_R.append(np.nan)
            pel_angle.append(np.nan)
            step_proj_L.append(np.nan); step_proj_R.append(np.nan)
            frame_flag.append(False)

            res = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = res.pose_landmarks
            if not lm:
                put_multiline(frame, "NO POSE", (20, 40), col=(0, 255, 255))
                writer.write(frame); pbar.update(1); continue

            pts = [(p.x, p.y, p.visibility) for p in lm.landmark]
            if min(pts[L_HIP][2], pts[R_HIP][2], pts[L_ANK][2], pts[R_ANK][2]) < 0.6:
                put_multiline(frame, "POSE VIS<0.6", (20, 40), col=(0, 255, 255))
                writer.write(frame); pbar.update(1); continue

            # 像素座標
            hip_L = np.array([pts[L_HIP][0] * W, pts[L_HIP][1] * H])
            hip_R = np.array([pts[R_HIP][0] * W, pts[R_HIP][1] * H])
            ank_L = np.array([pts[L_ANK][0] * W, pts[L_ANK][1] * H])
            ank_R = np.array([pts[R_ANK][0] * W, pts[R_ANK][1] * H])
            knee_L = np.array([pts[L_KNEE][0] * W, pts[L_KNEE][1] * H])
            knee_R = np.array([pts[R_KNEE][0] * W, pts[R_KNEE][1] * H])

            idx = -1  # 當前影格索引

            # 腿長
            LL = euclidean(pts[L_HIP], pts[L_ANK]) * W
            RL = euclidean(pts[R_HIP], pts[R_ANK]) * W
            len_L[idx] = LL; len_R[idx] = RL

            # 骨盆角度
            ang = math.degrees(math.atan2(abs(hip_L[1] - hip_R[1]),
                                          abs(hip_L[0] - hip_R[0]) + 1e-6))
            pel_angle[idx] = ang

            # 步長估計
            hip_mid_x = (hip_L[0] + hip_R[0]) / 2
            step_proj_L[idx] = abs(ank_L[0] - hip_mid_x)
            step_proj_R[idx] = abs(ank_R[0] - hip_mid_x)

            # ── 影格偵測旗標（三條件其一即可）
            mean_leg = (LL + RL) / 2
            diff_ratio = abs(LL - RL) / mean_leg
            frame_hit = (
                diff_ratio > LEN_DIFF_RATIO_TH or
                ang > PEL_ANGLE_TH_DEG
            )
            frame_flag[idx] = frame_hit

            # ── 視覺化 ──────────────────
            if frame_hit:
                short_left = LL < RL
                hip = hip_L if short_left else hip_R
                knee = knee_L if short_left else knee_R
                ank = ank_L if short_left else ank_R

                # 骨架線
                cv2.line(frame, hip.astype(int), knee.astype(int), (255, 255, 255), 2)
                cv2.line(frame, knee.astype(int), ank.astype(int), (255, 255, 255), 2)

                # 彩色點
                cv2.circle(frame, hip.astype(int), 8, (0, 165, 255), -1)  # 橙 Hip
                cv2.circle(frame, knee.astype(int), 8, (0, 255, 0), -1)   # 綠 Knee
                cv2.circle(frame, ank.astype(int), 8, (255, 0, 0), -1)    # 藍 Ankle

                leg_txt = "L" if short_left else "R"
                # ★ 將 fs 參數設定為放大倍率
                put_multiline(
                    frame,
                    f"LLD DETECTED\nShorter leg: {leg_txt}",
                    (20, 40),
                    fs=TEXT_SCALE_ALERT          # ★ 套用放大倍率
                )

            writer.write(frame); pbar.update(1)

    cap.release(); writer.release()

    # ── 統計 ───────────────────────────────
    arr_L = np.array(len_L); arr_R = np.array(len_R)
    if (~np.isnan(arr_L) & ~np.isnan(arr_R)).sum() < 10:
        return dict(LLD_flag=False, annotated_video=out_path, csv=None)

    mean_leg = (np.nanmean(arr_L) + np.nanmean(arr_R)) / 2
    diff_ratio = abs(np.nanmean(arr_L) - np.nanmean(arr_R)) / mean_leg
    pel_angle_mean = np.nanmean(pel_angle)
    stride_L = np.nanmax(step_proj_L) - np.nanmin(step_proj_L)
    stride_R = np.nanmax(step_proj_R) - np.nanmin(step_proj_R)
    step_diff_ratio = abs(stride_L - stride_R) / mean_leg

    overall_flag = (
        diff_ratio      > LEN_DIFF_RATIO_TH or
        pel_angle_mean  > PEL_ANGLE_TH_DEG  or
        step_diff_ratio > STEP_DIFF_RATIO_TH
    )

    csv_path = os.path.splitext(path)[0] + "_lld_series.csv"
    pd.DataFrame(dict(
        LL=len_L, RL=len_R, PelvisAngleDeg=pel_angle,
        StepProjL=step_proj_L, StepProjR=step_proj_R, FrameFlag=frame_flag
    )).to_csv(csv_path, index=False)

    return dict(
        mean_leg_L=np.nanmean(arr_L),
        mean_leg_R=np.nanmean(arr_R),
        length_diff_ratio=diff_ratio,
        pelvis_angle_deg=pel_angle_mean,
        step_diff_ratio=step_diff_ratio,
        LLD_flag=overall_flag,
        annotated_video=out_path,
        csv=csv_path,
    )

# ── CLI ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MediaPipe LLD detector – visual version")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    video = args.video
    out = args.output or os.path.splitext(video)[0] + "_annotated.mp4"

    try:
        res = analyse_and_annotate(video, out)
    except Exception as e:
        sys.exit(f"[Error] {e}")

    print("\n=== 影片分析結果 ===")
    for k, v in res.items():
        print(f"{k:20}: {v}")

if __name__ == "__main__":
    main()

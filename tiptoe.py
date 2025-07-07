#!/usr/bin/env python
# tiptoe_from_video_simple_cfg.py
# -----------------------------------------------------------
# 功能：以 Mediapipe Pose 偵測「踮腳尖走路（Tip-Toe Walking, TTW）」
# 作者：Yen 的 AI 助手（2025.07）
# -----------------------------------------------------------

# ========= ① 這裡改路徑即可 =========
VIDEO_PATH      = "C:\\Users\\User\\Desktop\\幼童_踮腳尖走路.mp4"
SAVE_ANNOTATION = True      # False → 僅輸出 JSON，不存標註影片
# ====================================

import cv2, mediapipe as mp, numpy as np, json, time, pathlib
from collections import deque

# ======== 可調參數 ========
PF_THRESHOLD_DEG     = 5.0   # 踝關節 > 5° 蹠屈 ⇒ 無 heel-strike
EARLY_RISE_PERCENT   = 0.30  # < 30 % gait cycle heel-rise ⇒ 異常
FRAME_RATIO_FLAG     = 0.50  # 全片 > 50 % 影格被標 TTW ⇒ 整段踮腳
WIN_SMOOTH_ANGLE     = 7     # 角度移動平均視窗(影格)
WIN_SMOOTH_ANKLE_Y   = 7     # 踝 y 移動平均視窗(影格)
FONT                = cv2.FONT_HERSHEY_SIMPLEX
# =========================

def angle(p1, p2, p3):
    """回傳 ∠p1-p2-p3（度），p = (x, y)。"""
    a, b, c = map(np.array, (p1, p2, p3))
    cosv = np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

def moving_avg(q: deque, val: float):
    q.append(val)
    return sum(q) / len(q)

def detect_tiptoe(video_path: pathlib.Path, save_video: bool):
    pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(str(video_path))
    fps, w, h = (cap.get(cv2.CAP_PROP_FPS) or 30,
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if save_video:
        out_path = video_path.with_stem(video_path.stem + "_ttw").with_suffix(".mp4")
        writer   = cv2.VideoWriter(str(out_path),
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    q_angle  = deque(maxlen=WIN_SMOOTH_ANGLE)
    q_ankleY = deque(maxlen=WIN_SMOOTH_ANKLE_Y)
    flags, heel_rise_events = [], 0
    gait_start, prev_ankle_y, idx = None, None, 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        is_tiptoe = False

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            RK, RA, RH, RT = lm[26], lm[28], lm[30], lm[32]  # 右膝踝踵趾

            ang  = angle((RK.x*w, RK.y*h), (RA.x*w, RA.y*h), (RT.x*w, RT.y*h))
            angS = moving_avg(q_angle, ang)

            q_ankleY.append(RA.y)
            ankleS = sum(q_ankleY)/len(q_ankleY)

            if angS > PF_THRESHOLD_DEG and RT.y < RH.y:
                is_tiptoe = True
                if save_video:
                    cv2.putText(frame, "TIPTOE", (50, 60), FONT, 1.8, (0, 0, 255), 3)

            # heel-rise 早期事件偵測
            if prev_ankle_y is not None:
                delta = ankleS - prev_ankle_y
                if delta > 0 and gait_start is None:     # ankle 開始下降
                    gait_start = idx
                if delta < 0 and gait_start is not None: # ankle 上升 → heel-rise
                    pct = (idx - gait_start) / fps
                    if pct < EARLY_RISE_PERCENT:
                        heel_rise_events += 1
                        if save_video:
                            cv2.putText(frame, "EARLY HEEL RISE", (50, 120),
                                        FONT, 1.2, (255, 0, 0), 2)
                    gait_start = None
            prev_ankle_y = ankleS

        flags.append(is_tiptoe)
        if save_video:
            writer.write(frame)
        idx += 1

    cap.release()
    if save_video:
        writer.release()
    pose.close()

    ratio = float(np.mean(flags))
    report = {
        "video": str(video_path),
        "fps": fps,
        "frames": idx,
        "tiptoe_frames": int(sum(flags)),
        "tiptoe_ratio": round(ratio, 3),
        "overall_tiptoe": ratio >= FRAME_RATIO_FLAG,
        "early_heel_rise_events": heel_rise_events
    }

    json_path = video_path.with_suffix(".ttw.json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # ---- 終端輸出 ----
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"⇢ 全程耗時：{time.time() - t0:.1f} 秒")
    if save_video:
        print(f"⇢ 已輸出標註影片：{out_path}")
    print(f"⇢ JSON 統計：{json_path}")

# ===== 程式進入點 =====
if __name__ == "__main__":
    detect_tiptoe(pathlib.Path(VIDEO_PATH), save_video=SAVE_ANNOTATION)

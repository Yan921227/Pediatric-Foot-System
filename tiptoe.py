#!/usr/bin/env python
# tiptoe_from_video_3d.py
# -----------------------------------------------------------
# 功能：以 Mediapipe Pose 3D 座標偵測「踮腳尖走路（TTW）」
# 作者：Yen 的 AI 助手（2025.07）
# -----------------------------------------------------------

# ========= ① 這裡改路徑即可 =========
VIDEO_PATH      = "C:\\Users\\User\\Desktop\\幼童_踮腳尖走路.mp4"
SAVE_ANNOTATION = True          # False → 僅輸出 JSON，不存標註影片
USE_3D          = True          # True → 用 3D 關節角；False → 回到 2D
# ====================================

import cv2, mediapipe as mp, numpy as np, json, time, pathlib
from collections import deque

# ======== 可調參數 ========
PF_THRESHOLD_DEG     = 8.0   # 3D 角度臨界值（2D 建議 5.0）
EARLY_RISE_PERCENT   = 0.30  # < 30 % gait cycle heel-rise ⇒ 異常
FRAME_RATIO_FLAG     = 0.50  # 全片 > 50 % 影格被標 TTW ⇒ 整段踮腳
WIN_SMOOTH_ANGLE     = 7     # 角度移動平均視窗(影格)
WIN_SMOOTH_ANKLE_Y   = 7     # 踝 y 移動平均視窗(影格)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# =========================

def angle2d(p1, p2, p3):
    """2D ∠p1-p2-p3（度）"""
    a, b, c = map(np.array, (p1, p2, p3))
    cosv = np.dot(a-b, c-b) / (np.linalg.norm(a-b)*np.linalg.norm(c-b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

def angle3d(p1, p2, p3):
    """3D ∠p1-p2-p3（度）"""
    return angle2d(p1, p2, p3) if not USE_3D else _angle3d_inner(p1, p2, p3)

def _angle3d_inner(p1, p2, p3):
    a, b, c = map(np.array, (p1, p2, p3))
    cosv = np.dot(a-b, c-b) / (np.linalg.norm(a-b)*np.linalg.norm(c-b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

def moving_avg(q: deque, val: float):
    q.append(val)
    return sum(q) / len(q)

def detect_tiptoe(video_path: pathlib.Path, save_video: bool):
    pose = mp.solutions.pose.Pose(
        model_complexity=1, smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(str(video_path))
    fps, w, h = (cap.get(cv2.CAP_PROP_FPS) or 30,
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if save_video:
        out_path = video_path.with_stem(video_path.stem + "_ttw").with_suffix(".mp4")
        writer   = cv2.VideoWriter(str(out_path),
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    q_angle, q_ankleY = deque(maxlen=WIN_SMOOTH_ANGLE), deque(maxlen=WIN_SMOOTH_ANKLE_Y)
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
            lm2d = res.pose_landmarks.landmark
            RK2, RA2, RH2, RT2 = lm2d[26], lm2d[28], lm2d[30], lm2d[32]

            # ------- 角度計算 -------
            if USE_3D and res.pose_world_landmarks:
                world = res.pose_world_landmarks.landmark
                RK, RA, RT = world[26], world[28], world[32]
                ang_raw = angle3d((RK.x, RK.y, RK.z),
                                  (RA.x, RA.y, RA.z),
                                  (RT.x, RT.y, RT.z))
            else:  # 回退 2D
                ang_raw = angle2d((RK2.x*w, RK2.y*h),
                                  (RA2.x*w, RA2.y*h),
                                  (RT2.x*w, RT2.y*h))

            ang_sm = moving_avg(q_angle, ang_raw)

            # ------- 踝 y 平滑 -------
            q_ankleY.append(RA2.y)
            ankle_sm = sum(q_ankleY) / len(q_ankleY)

            # ------- 判斷踮腳 -------
            toe_higher = RT2.y < RH2.y
            if ang_sm > PF_THRESHOLD_DEG and toe_higher:
                is_tiptoe = True
                if save_video:
                    cv2.putText(frame, "TIPTOE", (50, 60), FONT, 1.8, (0, 0, 255), 3)

            # ------- early heel-rise -------
            if prev_ankle_y is not None:
                delta = ankle_sm - prev_ankle_y
                if delta > 0 and gait_start is None:     # ankle 開始下降
                    gait_start = idx
                if delta < 0 and gait_start is not None: # ankle 上升 → heel-rise
                    pct = (idx - gait_start) / fps
                    if pct < EARLY_RISE_PERCENT:
                        heel_rise_events += 1
                        if save_video:
                            cv2.putText(frame, "EARLY HEEL RISE",
                                        (50, 120), FONT, 1.2, (255, 0, 0), 2)
                    gait_start = None
            prev_ankle_y = ankle_sm

        flags.append(is_tiptoe)
        if save_video:
            writer.write(frame)
        idx += 1

    cap.release()
    if save_video:
        writer.release()
    pose.close()

    ratio = float(np.mean(flags)) if flags else 0.0
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

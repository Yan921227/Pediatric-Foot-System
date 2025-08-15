#!/usr/bin/env python
# ttw_inframe_fixed_block.py
# -----------------------------------------------------------
# 在「影片內固定區塊(HUD)」顯示角度；僅在該腳判定為踮腳時顯示數值
# 判斷：∠(膝/跟, 踝, 第1趾) > 門檻 且 toe_y < heel_y
# 無 TIPTOE 大字、無 EHR、無 PyQt；正常速度播放
# -----------------------------------------------------------

# ========= ① 這裡改路徑即可 =========
VIDEO_PATH        = r"D:\幼童足部辨識\踮腳走路\幼童_踮腳尖走路.mp4"
USE_3D            = False      # 有 world_landmarks 就用 3D，否則回退 2D
SAVE_ANNOTATION   = True       # ← True 才會輸出含標註影片
OUTPUT_DIR        = None       # None=與來源同資料夾；或填 r"D:\export"
OUTPUT_SUFFIX     = "_ttw"     # 會輸出成 <原檔名>_ttw.mp4
AVOID_OVERWRITE   = True       # True：若重名會自動加 _1, _2 ...
OUTPUT_CODEC      = "mp4v"     # 常見：mp4v（.mp4），或 "XVID"（.avi）
PLAYBACK_SPEED    = 1.0        # 播放速度
SHOW_ANGLE_DEF    = False      # HUD 是否顯示角度定義
# ====================================

import time
import numpy as np
import cv2, mediapipe as mp
from collections import deque
from pathlib import Path

# ====== 顯示/濾波參數 ======
PF_THRESHOLD_DEG   = 8.0
SMA_WIN            = 7
EMA2D_ALPHA        = 0.35
EMA3D_ALPHA        = 0.35

# === 角度定義切換 ===
# 'KAT' = knee-ankle-toe（膝-踝-趾；原本做法）
# 'HAT' = heel-ankle-toe（跟-踝-趾；較不受膝擺動影響）
ANGLE_DEF         = 'KAT'

# HUD 固定區塊
PANEL_POS          = "tr"      # 'tr','tl','br','bl'
PANEL_W_RATIO      = 0.50
PANEL_ALPHA        = 0.75
PANEL_BG_COLOR     = (255, 255, 255)  # BGR
FONT               = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE_REF     = 0.60
LINE_H_REF         = 28
TEXT_COLOR         = (255, 0, 0)      # BGR
TEXT_THICK         = 1

# 關鍵點簡單標示
DOT_RADIUS         = 4
DOT_COLOR          = (0, 200, 0)
SEG_COLOR          = (180, 180, 180)

mp_pose = mp.solutions.pose

# --------- 小工具 ----------
class SMA:
    def __init__(self, win=7): self.q = deque(maxlen=win)
    def push(self, v):
        self.q.append(float(v));  return sum(self.q)/len(self.q)

class EMA:
    def __init__(self, alpha=0.3): self.alpha=float(alpha); self.v=None
    def push(self, x):
        x = np.asarray(x, dtype=float)
        self.v = x if self.v is None else self.alpha*x + (1-self.alpha)*self.v
        return self.v

def angle_nd(a, b, c):
    a,b,c = map(np.asarray,(a,b,c))
    v1, v2 = a-b, c-b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosv  = float(np.dot(v1, v2) / denom)
    return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))

def fmt_num(val, digits=2):
    if val is None or np.isnan(val) or np.isinf(val):  return "N/A"
    return f"{float(val):.{digits}f}"

def build_out_path(video_path, suffix="_ttw", ext=".mp4",
                   out_dir=None, avoid_overwrite=True):
    p = Path(video_path)
    out_dir = Path(out_dir) if out_dir else p.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = p.stem + suffix
    out_path = out_dir / f"{stem}{ext}"
    if avoid_overwrite:
        k, final = 1, out_path
        while final.exists():
            final = out_dir / f"{stem}_{k}{ext}"
            k += 1
        out_path = final
    return out_path

def draw_info_panel(frame, lines, pos="tr"):
    """在 frame 上畫固定大小的 HUD（不隨內容變大）"""
    h, w = frame.shape[:2]
    panel_w   = int(w * PANEL_W_RATIO)
    font_scale = FONT_SCALE_REF * (h/1080.0)
    line_h     = int(max(18, LINE_H_REF * (h/1080.0)))
    pad        = int(12 * (h/1080.0))
    box_h      = pad*2 + line_h*len(lines)

    if pos == "tr":   x0, y0 = w - panel_w - 10, 10
    elif pos == "tl": x0, y0 = 10, 10
    elif pos == "br": x0, y0 = w - panel_w - 10, h - box_h - 10
    else:             x0, y0 = 10, h - box_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+box_h), PANEL_BG_COLOR, -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1-PANEL_ALPHA, 0, frame)

    y = y0 + pad + line_h - int(line_h*0.3)
    for text in lines:
        cv2.putText(frame, text, (x0+pad, y), FONT, font_scale, TEXT_COLOR, TEXT_THICK, cv2.LINE_AA)
        y += line_h

def draw_side(frame, get2d, K, A, T, H):
    xK,yK = map(int, get2d(K)); xA,yA = map(int, get2d(A))
    xT,yT = map(int, get2d(T)); xH,yH = map(int, get2d(H))
    cv2.circle(frame,(xK,yK),DOT_RADIUS,DOT_COLOR,-1)
    cv2.circle(frame,(xA,yA),DOT_RADIUS,DOT_COLOR,-1)
    cv2.circle(frame,(xT,yT),DOT_RADIUS,DOT_COLOR,-1)
    cv2.circle(frame,(xH,yH),DOT_RADIUS,DOT_COLOR,-1)
    cv2.line(frame,(xK,yK),(xA,yA),SEG_COLOR,2)
    cv2.line(frame,(xA,yA),(xT,yT),SEG_COLOR,2)
    return xA,yA

def ankle_def_desc():
    return "knee-ankle-toe" if ANGLE_DEF.upper() == "KAT" else "heel-ankle-toe"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[!] 無法開啟影片：{VIDEO_PATH}"); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_dt = (1.0 / fps) / max(PLAYBACK_SPEED, 1e-6)
    next_t = time.perf_counter() + target_dt

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ===== 建立輸出 writer（若啟用） =====
    writer, out_path = None, None
    if SAVE_ANNOTATION:
        ext = ".mp4" if OUTPUT_CODEC.lower() == "mp4v" else ".avi"
        out_path = build_out_path(
            VIDEO_PATH, suffix=OUTPUT_SUFFIX, ext=ext,
            out_dir=OUTPUT_DIR, avoid_overwrite=AVOID_OVERWRITE
        )
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            print("[!] 指定的編碼器無法開啟，改用 XVID(.avi) 備援。")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_path = build_out_path(VIDEO_PATH, suffix=OUTPUT_SUFFIX, ext=".avi",
                                      out_dir=OUTPUT_DIR, avoid_overwrite=AVOID_OVERWRITE)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"[i] 會輸出標註影片：{out_path}")
        else:
            print("[!] 無法建立輸出影片，將只顯示不寫檔。")
            writer = None

    pose = mp_pose.Pose(model_complexity=2, smooth_landmarks=True,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6)

    # ===== 建立視窗（無標題文字） =====
    WIN_NAME = " "  # 單一空白：標題列看起來沒有文字
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)
    # 兼容舊版：若支援 setWindowTitle，強制把標題設為空白
    try:
        cv2.setWindowTitle(WIN_NAME, " ")
    except Exception:
        pass

    # MediaPipe 索引
    L_KNEE, L_ANK, L_HEEL, L_TOE = 25, 27, 29, 31
    R_KNEE, R_ANK, R_HEEL, R_TOE = 26, 28, 30, 32

    # 濾波器
    sma_pf = {"L": SMA(SMA_WIN), "R": SMA(SMA_WIN)}
    ema2d  = {i: EMA(EMA2D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}
    ema3d  = {i: EMA(EMA3D_ALPHA) for i in (25,27,29,31, 26,28,30,32)}

    frame_count = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_count += 1

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        L_show = R_show = None

        if res.pose_landmarks:
            lm2d = res.pose_landmarks.landmark
            use_world = USE_3D and (res.pose_world_landmarks is not None)
            world = res.pose_world_landmarks.landmark if use_world else None

            def get2d(i):
                u = lm2d[i]; pt = np.array([u.x*w, u.y*h], float)
                return ema2d[i].push(pt)

            def get3d(i):
                if world is None:
                    x,y = get2d(i); pt = np.array([x, y, 0.0], float)
                else:
                    u = world[i];   pt = np.array([u.x, u.y, u.z], float)
                return ema3d[i].push(pt)

            # === 依定義計算踝角 ===
            if ANGLE_DEF.upper() == 'KAT':
                L_raw = angle_nd(get3d(L_KNEE), get3d(L_ANK), get3d(L_TOE))
                R_raw = angle_nd(get3d(R_KNEE), get3d(R_ANK), get3d(R_TOE))
            else:  # 'HAT'
                L_raw = angle_nd(get3d(L_HEEL), get3d(L_ANK), get3d(L_TOE))
                R_raw = angle_nd(get3d(R_HEEL), get3d(R_ANK), get3d(R_TOE))

            L_pf = sma_pf["L"].push(L_raw)
            R_pf = sma_pf["R"].push(R_raw)

            # 踮腳判斷（以 2D y 判斷趾高於跟；y 越小越高）
            left_tiptoe  = (L_pf > PF_THRESHOLD_DEG) and (lm2d[L_TOE].y < lm2d[L_HEEL].y)
            right_tiptoe = (R_pf > PF_THRESHOLD_DEG) and (lm2d[R_TOE].y < lm2d[R_HEEL].y)

            # 畫最小關鍵點骨架（兩側都畫）
            draw_side(frame, get2d, L_KNEE, L_ANK, L_TOE, L_HEEL)
            draw_side(frame, get2d, R_KNEE, R_ANK, R_TOE, R_HEEL)

            # 僅在踮腳時，把數值放到 HUD
            if left_tiptoe:  L_show = float(L_pf)
            if right_tiptoe: R_show = float(R_pf)

        # ---- HUD ----
        base = f"ankle angle ({ankle_def_desc()})" if SHOW_ANGLE_DEF else "ankle angle"
        lines = [f"Left ankle angle:  {fmt_num(L_show)} deg",
                 f"Right ankle angle: {fmt_num(R_show)} deg"]
        draw_info_panel(frame, lines, pos=PANEL_POS)

        # 顯示 / 存檔
        cv2.imshow(WIN_NAME, frame)
        if writer is not None: writer.write(frame)

        # 正常速度節拍
        now = time.perf_counter()
        sleep_s = next_t - now
        if sleep_s > 0: time.sleep(sleep_s)
        next_t += target_dt

        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC 離開
            break

    # ===== 清理與落款 =====
    cap.release()
    pose.close()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    dt = time.perf_counter() - t0
    if SAVE_ANNOTATION and out_path and Path(out_path).exists():
        print(f"[✓] 輸出完成：{out_path}")
        print(f"[i] 解析度：{w}x{h}，幀數：{frame_count}，FPS：{fps:.2f}，耗時：{dt:.2f}s")
    else:
        print("[i] 沒有輸出影片（SAVE_ANNOTATION=False 或 writer 建立失敗）。")

if __name__ == "__main__":
    main()

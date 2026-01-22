import cv2
import mediapipe as mp
import numpy as np

# ======================
# 使用者設定
# ======================
VIDEO_PATH = "C:\\Users\\User\\Desktop\\walk_behind.mp4"
OUTPUT_PATH = "output_resultghum 3d.mp4"

SWAY_THRESHOLD_CM = 5.0       # 單腳站立搖晃閾值（根據圖二標準：>5cm）

# ======================
# MediaPipe 初始化
# ======================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
drawing_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=3)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ======================
# 工具函式
# ======================
def dist(a, b):
    return np.linalg.norm(a - b)

# ======================
# 影片設定
# ======================
cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

ankle_points = []
step_widths = []

# ======================
# 主迴圈
# ======================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks and res.pose_world_landmarks:
        # 繪製骨架（使用 pose_landmarks 用於圖像顯示）
        mp_draw.draw_landmarks(
            frame, 
            res.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

        # 獲取真實世界3D座標（使用 pose_world_landmarks，單位：米）
        lm_world = res.pose_world_landmarks.landmark
        
        ra = lm_world[mp_pose.PoseLandmark.RIGHT_ANKLE]
        la = lm_world[mp_pose.PoseLandmark.LEFT_ANKLE]
        rk = lm_world[mp_pose.PoseLandmark.RIGHT_KNEE]
        lk = lm_world[mp_pose.PoseLandmark.LEFT_KNEE]

        # 真實世界3D座標（米）
        ra_3d = np.array([ra.x, ra.y, ra.z])
        la_3d = np.array([la.x, la.y, la.z])
        rk_3d = np.array([rk.x, rk.y, rk.z])
        lk_3d = np.array([lk.x, lk.y, lk.z])

        ankle_points.append(ra_3d)

        # 步寬（水平距離，米）
        step_width = np.linalg.norm(ra_3d[[0, 2]] - la_3d[[0, 2]])
        step_widths.append(step_width)

        # 即時計算並顯示當前狀態
        if len(ankle_points) > 1:
            current_ankle_points = np.array(ankle_points)
            horizontal_disp_now = np.ptp(current_ankle_points[:, [0, 2]], axis=0)
            sway_m_now = np.linalg.norm(horizontal_disp_now)  # 米
            sway_cm_now = sway_m_now * 100  # 轉換成公分
            
            avg_step_width_now = np.mean(step_widths) if step_widths else 0
            step_width_cm_now = avg_step_width_now * 100  # 轉換成公分
            
            # 計算小腿長度（用於參考）
            shank_avg = (dist(rk_3d, ra_3d) + dist(lk_3d, la_3d)) / 2
            step_width_ratio_now = avg_step_width_now / shank_avg if shank_avg > 0 else 0
            
            # 即時判斷（根據圖二標準：單腳站立搖晃 > 5cm）
            sway_fail_now = sway_cm_now > SWAY_THRESHOLD_CM
            
            if sway_fail_now:
                result_now = "Flat Foot Detected"
                result_color = (0, 0, 255)  # 紅色
            else:
                result_now = "Normal Walking"
                result_color = (0, 255, 0)  # 綠色
            
            # 顯示資訊面板（半透明背景）
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (450, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # 顯示即時數據
            cv2.putText(frame, "=== Real-time Analysis ===", (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Sway: {sway_cm_now:.1f} cm", (15, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(frame, f"(Threshold: >{SWAY_THRESHOLD_CM} cm)", (250, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            cv2.putText(frame, f"Step Width: {step_width_cm_now:.1f} cm", (15, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            
            cv2.putText(frame, f"(Reference only)", (250, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            
            # 顯示即時判斷結果
            cv2.putText(frame, f"Status: {result_now}", (15, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)

    writer.write(frame)
    cv2.imshow("Processing", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

# ======================
# 最終統計
# ======================
if len(ankle_points) > 0:
    ankle_points = np.array(ankle_points)
    horizontal_disp = np.ptp(ankle_points[:, [0, 2]], axis=0)
    sway_m = np.linalg.norm(horizontal_disp)  # 米
    sway_cm = sway_m * 100  # 轉換成公分
    
    avg_step_width = np.mean(step_widths)
    step_width_cm = avg_step_width * 100  # 轉換成公分
    
    # 計算步寬比例（僅供參考）
    if len(step_widths) > 0:
        step_width_ratio = step_width_cm / 30.0  # 假設平均小腿30cm作為參考
    else:
        step_width_ratio = 0
    
    # 根據圖二標準判斷：單腳站立搖晃 > 5cm
    sway_fail = sway_cm > SWAY_THRESHOLD_CM
    
    if sway_fail:
        result = "Flat Foot (扁平足 - 單腳站立搖晃超過閾值)"
    else:
        result = "Normal (正常 - 單腳站立搖晃在正常範圍)"
    
    print("==== 分析完成 ====")
    print(f"單腳站立搖晃: {sway_cm:.2f} cm (閾值: >{SWAY_THRESHOLD_CM} cm) - {'異常' if sway_fail else '正常'}")
    print(f"步寬(參考): {step_width_cm:.2f} cm (比例: {step_width_ratio:.2f})")
    print(f"判斷結果: {result}")
    print(f"輸出影片: {OUTPUT_PATH}")
else:
    print("警告：未偵測到足夠的姿態數據")

import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 定義字體
FONT = cv2.FONT_HERSHEY_COMPLEX

def get_leg_points(landmarks):
    """提取腿部關鍵點座標"""
    if not landmarks:
        return None
    
    # MediaPipe pose landmarks 索引
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    points = {}
    try:
        points['left_hip'] = [landmarks.landmark[LEFT_HIP].x, landmarks.landmark[LEFT_HIP].y]
        points['right_hip'] = [landmarks.landmark[RIGHT_HIP].x, landmarks.landmark[RIGHT_HIP].y]
        points['left_knee'] = [landmarks.landmark[LEFT_KNEE].x, landmarks.landmark[LEFT_KNEE].y]
        points['right_knee'] = [landmarks.landmark[RIGHT_KNEE].x, landmarks.landmark[RIGHT_KNEE].y]
        points['left_ankle'] = [landmarks.landmark[LEFT_ANKLE].x, landmarks.landmark[LEFT_ANKLE].y]
        points['right_ankle'] = [landmarks.landmark[RIGHT_ANKLE].x, landmarks.landmark[RIGHT_ANKLE].y]
        
        return points
    except:
        return None

def calculate_distance_in_pixels(point1, point2, img_width, img_height):
    """計算兩點間的像素距離"""
    x1, y1 = point1[0] * img_width, point1[1] * img_height
    x2, y2 = point2[0] * img_width, point2[1] * img_height
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def estimate_body_height_pixels(points, img_width, img_height):
    """估算身體高度（像素）- 從髖關節到踝關節"""
    # 計算左右腿的平均長度作為參考
    left_leg_length = calculate_distance_in_pixels(points['left_hip'], points['left_ankle'], img_width, img_height)
    right_leg_length = calculate_distance_in_pixels(points['right_hip'], points['right_ankle'], img_width, img_height)
    avg_leg_length = (left_leg_length + right_leg_length) / 2
    
    # 假設腿長約為身高的50%，推估身高
    estimated_height_pixels = avg_leg_length * 2
    return estimated_height_pixels

def pixels_to_cm(distance_pixels, estimated_height_pixels, assumed_height_cm=150):
    """將像素距離轉換為公分（基於假設的身高）"""
    # 預設假設身高150公分，可根據實際情況調整
    cm_per_pixel = assumed_height_cm / estimated_height_pixels
    return distance_pixels * cm_per_pixel

def determine_leg_type(points, img_width, img_height):
    """判斷腿型 - 基於醫學標準距離測量，X型腿與O型腿分開判斷"""
    
    # 計算關鍵距離（像素）
    knee_distance_pixels = calculate_distance_in_pixels(points['left_knee'], points['right_knee'], img_width, img_height)
    ankle_distance_pixels = calculate_distance_in_pixels(points['left_ankle'], points['right_ankle'], img_width, img_height)
    
    # 估算身高並轉換為公分
    estimated_height_pixels = estimate_body_height_pixels(points, img_width, img_height)
    knee_distance_cm = pixels_to_cm(knee_distance_pixels, estimated_height_pixels)
    ankle_distance_cm = pixels_to_cm(ankle_distance_pixels, estimated_height_pixels)
    
    # 醫學標準閾值
    O_LEG_KNEE_THRESHOLD = 6.0  # O型腿：膝蓋間距 > 6公分
    X_LEG_ANKLE_THRESHOLD = 8.0  # X型腿：踝關節間距 > 8公分
    
    # 分別判斷X型腿和O型腿
    result = {
        'knee_distance_cm': knee_distance_cm,
        'ankle_distance_cm': ankle_distance_cm,
        'knee_distance_pixels': knee_distance_pixels,
        'ankle_distance_pixels': ankle_distance_pixels
    }
    
    # O型腿判斷
    if knee_distance_cm > O_LEG_KNEE_THRESHOLD:
        result['o_leg'] = True
        result['o_leg_severity'] = "true" if knee_distance_cm > O_LEG_KNEE_THRESHOLD * 1.5 else "mild"
    else:
        result['o_leg'] = False
        result['o_leg_severity'] = "false"
    
    # X型腿判斷
    if ankle_distance_cm > X_LEG_ANKLE_THRESHOLD:
        result['x_leg'] = True
        result['x_leg_severity'] = "true" if ankle_distance_cm > X_LEG_ANKLE_THRESHOLD * 1.5 else "mild"
    else:
        result['x_leg'] = False
        result['x_leg_severity'] = "false"
    
    return result

def draw_leg_analysis(img, points, analysis_result):
    """在圖像上繪製分析結果"""
    h, w = img.shape[:2]
    
    # 繪製關鍵點
    point_colors = {
        'left_hip': (255, 0, 0), 'right_hip': (255, 0, 0),
        'left_knee': (0, 255, 0), 'right_knee': (0, 255, 0),
        'left_ankle': (0, 0, 255), 'right_ankle': (0, 0, 255)
    }
    
    for key, point in points.items():
        x, y = int(point[0] * w), int(point[1] * h)
        color = point_colors.get(key, (255, 255, 255))
        cv2.circle(img, (x, y), 6, color, -1)
    
    # 繪製連線
    def draw_line(p1_key, p2_key, color=(255, 255, 255), thickness=2):
        p1 = points[p1_key]
        p2 = points[p2_key]
        cv2.line(img, 
                (int(p1[0] * w), int(p1[1] * h)), 
                (int(p2[0] * w), int(p2[1] * h)), 
                color, thickness)
    
    # 繪製腿部骨架
    draw_line('left_hip', 'left_knee', (255, 0, 0), 2)    # 左大腿
    draw_line('left_knee', 'left_ankle', (255, 0, 0), 2)  # 左小腿
    draw_line('right_hip', 'right_knee', (0, 0, 255), 2)  # 右大腿
    draw_line('right_knee', 'right_ankle', (0, 0, 255), 2)# 右小腿
    
    # 繪製測量線
    draw_line('left_knee', 'right_knee', (0, 255, 255), 3)     # 膝蓋間距
    draw_line('left_ankle', 'right_ankle', (255, 255, 0), 3)   # 踝關節間距
    
    # 準備文字內容
    if analysis_result['o_leg']:
        o_text = f"O-shaped legs: {analysis_result['o_leg_severity']} ({analysis_result['knee_distance_cm']:.1f}cm)"
        o_color = (0, 0, 255)
    else:
        o_text = f"O-shaped legs: normal ({analysis_result['knee_distance_cm']:.1f}cm)"
        o_color = (0, 255, 0)

    if analysis_result['x_leg']:
        x_text = f"X-shaped legs: {analysis_result['x_leg_severity']} ({analysis_result['ankle_distance_cm']:.1f}cm)"
        x_color = (0, 0, 255)
    else:
        x_text = f"X-shaped legs: normal ({analysis_result['ankle_distance_cm']:.1f}cm)"
        x_color = (0, 255, 0)
    
    # 計算文字大小和位置
    font_scale = 0.6
    thickness = 2
    
    # 獲取文字尺寸
    (o_text_width, o_text_height), _ = cv2.getTextSize(o_text, FONT, font_scale, thickness)
    (x_text_width, x_text_height), _ = cv2.getTextSize(x_text, FONT, font_scale, thickness)
    
    # 計算背景框尺寸
    max_text_width = max(o_text_width, x_text_width)
    total_height = o_text_height + x_text_height + 30  # 30是間距
    box_width = max_text_width + 20  # 左右各10像素邊距
    box_height = total_height + 10   # 上下各5像素邊距
    
    # 繪製白色背景
    cv2.rectangle(img, (5, 5), (5 + box_width, 5 + box_height), (255, 255, 255), -1)
    
    # 顯示診斷結果（在白色背景上）
    y_offset = 5 + o_text_height + 5  # 5像素上邊距 + 文字高度 + 5像素間距
    
    # O型腿診斷
    cv2.putText(img, o_text, (15, y_offset), FONT, font_scale, o_color, thickness)
    
    y_offset += x_text_height + 15  # 文字高度 + 15像素間距
    
    # X型腿診斷
    cv2.putText(img, x_text, (15, y_offset), FONT, font_scale, x_color, thickness)

def draw_measurement_guide(img):
    """繪製測量指導"""
    pass  # 移除所有提示文字以保持畫面整潔

# 主程式 - 影片處理版本
def process_video(input_path, output_path):
    """處理影片並輸出辨識結果"""
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"無法開啟影片: {input_path}")
        return
    
    # 獲取影片資訊
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 設定影片編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"開始處理影片...")
    print(f"原始尺寸: {width}x{height}, FPS: {fps}, 總幀數: {total_frames}")
    print(f"輸出尺寸: {width}x{height}")
    
    frame_count = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as holistic:
        
        while True:
            ret, img = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # 每30幀顯示一次進度
                print(f"處理進度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # img = cv2.resize(img, (520, 300))  # 保留原始比例與解析度，不做強制縮放
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img2)
            
            # 繪製身體骨架（淡化顯示）
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1))
            
            # 腿型分析
            if results.pose_landmarks:
                leg_points = get_leg_points(results.pose_landmarks)
                if leg_points:
                    h, w = img.shape[:2]
                    analysis_result = determine_leg_type(leg_points, w, h)
                    draw_leg_analysis(img, leg_points, analysis_result)
            
            # 寫入輸出影片
            out.write(img)
    
    cap.release()
    out.release()
    print(f"影片處理完成！輸出檔案: {output_path}")

# 使用範例
if __name__ == "__main__":
    # 設定輸入和輸出路徑
    input_video_path = "D:\\幼童足部辨識\\xo\\xo.mp4"  # 請修改為你的輸入影片路徑
    output_video_path = "D:\\幼童足部辨識\\xo\\output_video.mp4"  # 輸出影片路徑

    # 處理影片
    process_video(input_video_path, output_video_path)
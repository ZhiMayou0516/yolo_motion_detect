import cv2
import numpy as np
import time
from ultralytics import YOLO


# ==========================================
# 1. 核心计算与平滑滤波
# ==========================================
def calculate_angle(a, b, c):
    """计算 2D 夹角"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.all(a == 0) or np.all(b == 0) or np.all(c == 0): return 0.0
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def calc_velocity(curr_pos, prev_pos, dt):
    """计算速度"""
    if prev_pos is None or dt <= 0 or np.all(curr_pos == 0): return 0.0
    dist = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
    return dist / dt


def calc_angular_velocity(curr_angle, prev_angle, dt):
    """计算角速度"""
    if prev_angle is None or dt <= 0 or curr_angle == 0: return 0.0
    return abs(curr_angle - prev_angle) / dt


def smooth_data(new_val, old_val, alpha=0.3):
    """指数移动平均滤波 (消除数据毛刺突变)"""
    if old_val is None or old_val == 0: return new_val
    return alpha * new_val + (1 - alpha) * old_val


# ==========================================
# 2. 主程序：YOLO推理与画面参数叠加
# ==========================================
def main():
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(1)

    # 用于计算和平滑的历史状态
    prev_time = time.time()
    prev_pos = {'Wrist': None, 'Ankle': None}
    prev_ang = {'Shoulder': None, 'Elbow': None, 'Knee': None}
    smooth_v = {'Wrist': 0, 'Ankle': 0}
    smooth_w = {'Shoulder': 0, 'Elbow': 0, 'Knee': 0}

    print(">>> 正在启动实时姿态与运动学参数检测... (按 'q' 退出)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        curr_time = time.time()
        dt = max(curr_time - prev_time, 0.001)

        results = model(frame, stream=True, verbose=False)
        annotated_frame = frame.copy()

        for r in results:
            # 绘制骨骼外框和关键点连线
            annotated_frame = r.plot()

            if r.keypoints is not None and len(r.keypoints.xy) > 0:

                # =========================
                # 给每个检测到的人标 ID
                # =========================
                for person_idx, kp_tensor in enumerate(r.keypoints.xy):
                    kp_person = kp_tensor.cpu().numpy()

                    # 优先用检测框左上角标 ID，比用鼻子更稳定
                    if r.boxes is not None and len(r.boxes.xyxy) > person_idx:
                        box = r.boxes.xyxy[person_idx].cpu().numpy()
                        x1, y1, x2, y2 = box.astype(int)

                        cv2.putText(
                            annotated_frame,
                            f"ID:{person_idx}",
                            (x1, max(y1 - 10, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2
                        )

                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )

                    else:
                        # 如果没有检测框，就退而求其次用鼻子位置
                        nose = kp_person[0]
                        if not np.all(nose == 0):
                            x, y = int(nose[0]), int(nose[1])
                            cv2.putText(
                                annotated_frame,
                                f"ID:{person_idx}",
                                (x, max(y - 10, 30)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 0, 255),
                                2
                            )

                # =========================
                # 仍然只取 ID:0 这个人进行角度计算
                # =========================
                kp = r.keypoints.xy[0].cpu().numpy()

                if len(kp) >= 16:
                    l_shoulder, l_elbow, l_wrist = kp[5], kp[7], kp[9]
                    l_hip, l_knee, l_ankle = kp[11], kp[13], kp[15]

                    # 【参数 1-3：关节角度】
                    ang_s = calculate_angle(l_hip, l_shoulder, l_elbow)
                    ang_e = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    ang_k = calculate_angle(l_hip, l_knee, l_ankle)

                    # 【参数 4-5：端点速度 (含滤波)】
                    raw_v_wrist = calc_velocity(l_wrist, prev_pos['Wrist'], dt)
                    raw_v_ankle = calc_velocity(l_ankle, prev_pos['Ankle'], dt)
                    smooth_v['Wrist'] = smooth_data(raw_v_wrist, smooth_v['Wrist'])
                    smooth_v['Ankle'] = smooth_data(raw_v_ankle, smooth_v['Ankle'])

                    # 【参数 6-8：角速度 (含滤波)】
                    raw_w_s = calc_angular_velocity(ang_s, prev_ang['Shoulder'], dt)
                    raw_w_e = calc_angular_velocity(ang_e, prev_ang['Elbow'], dt)
                    raw_w_k = calc_angular_velocity(ang_k, prev_ang['Knee'], dt)
                    smooth_w['Shoulder'] = smooth_data(raw_w_s, smooth_w['Shoulder'])
                    smooth_w['Elbow'] = smooth_data(raw_w_e, smooth_w['Elbow'])
                    smooth_w['Knee'] = smooth_data(raw_w_k, smooth_w['Knee'])

                    # 更新历史状态
                    prev_pos['Wrist'] = l_wrist
                    prev_pos['Ankle'] = l_ankle
                    prev_ang['Shoulder'] = ang_s
                    prev_ang['Elbow'] = ang_e
                    prev_ang['Knee'] = ang_k

                    # ==========================================
                    # 将参数绘制在画面左上角
                    # ==========================================
                    # 1. 绘制半透明黑色底板以增加文字可读性
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (320, 310), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

                    # 2. 准备要显示的文字内容
                    info_texts = [
                        "[Joint Angles] (Deg)",
                        f" Shoulder: {ang_s:.1f}",
                        f" Elbow: {ang_e:.1f}",
                        f" Knee: {ang_k:.1f}",
                        "",
                        "[Endpoint Velocity] (px/s)",
                        f" Wrist: {smooth_v['Wrist']:.1f}",
                        f" Ankle: {smooth_v['Ankle']:.1f}",
                        "",
                        "[Angular Velocity] (Deg/s)",
                        f" Shoulder: {smooth_w['Shoulder']:.1f}",
                        f" Elbow: {smooth_w['Elbow']:.1f}",
                        f" Knee: {smooth_w['Knee']:.1f}"
                    ]

                    # 3. 循环逐行打印文字
                    y_offset = 35
                    for text in info_texts:
                        if text == "":  # 如果是空字符串，减少一点行距
                            y_offset += 10
                            continue

                        # 区分标题和数值的颜色 (标题黄色，数值绿色)
                        color = (0, 255, 255) if text.startswith("[") else (0, 255, 0)
                        cv2.putText(annotated_frame, text, (20, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 22

        cv2.imshow('Camera - Pose & Kinematics', annotated_frame)
        prev_time = curr_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
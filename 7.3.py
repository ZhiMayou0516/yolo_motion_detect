import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode, OBFormat


# ============================================================
# 1. COCO 17个关键点定义
# ============================================================
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

DISPLAY_NAMES = {
    "nose": "Nose",
    "left_eye": "L_Eye",
    "right_eye": "R_Eye",
    "left_ear": "L_Ear",
    "right_ear": "R_Ear",
    "left_shoulder": "L_Sho",
    "right_shoulder": "R_Sho",
    "left_elbow": "L_Elb",
    "right_elbow": "R_Elb",
    "left_wrist": "L_Wri",
    "right_wrist": "R_Wri",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_ankle": "L_Ank",
    "right_ankle": "R_Ank"
}


# ============================================================
# 2. 深度与3D坐标计算函数
# ============================================================
def get_valid_depth_mm(depth_matrix, u, v, depth_scale=1.0, window_size=5):
    """
    在关键点附近取一个小窗口，对非0深度取中值。
    返回单位：mm
    """
    h, w = depth_matrix.shape[:2]

    u = int(round(u))
    v = int(round(v))

    if u < 0 or u >= w or v < 0 or v >= h:
        return None

    half = window_size // 2
    x1 = max(0, u - half)
    x2 = min(w, u + half + 1)
    y1 = max(0, v - half)
    y2 = min(h, v + half + 1)

    patch = depth_matrix[y1:y2, x1:x2].astype(np.float32)
    valid = patch[patch > 0]

    if valid.size == 0:
        return None

    depth_raw = np.median(valid)
    depth_mm = depth_raw * depth_scale

    return float(depth_mm)


def pixel_to_camera_3d(u, v, depth_mm, fx, fy, cx, cy):
    """
    像素坐标 + 深度值 -> 相机坐标系3D坐标
    输出单位：mm
    """
    z = depth_mm
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return float(x), float(y), float(z)


# ============================================================
# 3. 图像解码函数
# ============================================================
def decode_color_frame(color_frame):
    """
    将 Orbbec color_frame 解码为 OpenCV BGR 图像。
    """
    color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if color_frame.get_format() == OBFormat.MJPG:
        color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
    else:
        color_image = color_data.reshape(
            (color_frame.get_height(), color_frame.get_width(), 3)
        )
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    return color_image


# ============================================================
# 4. 在关键点旁边绘制3D坐标
# ============================================================
def draw_text_with_background(
    image,
    text,
    x,
    y,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.38,
    text_color=(0, 255, 255),
    bg_color=(0, 0, 0),
    thickness=1
):
    """
    给文字加黑色背景，避免文字看不清。
    """
    h, w = image.shape[:2]

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x = int(x)
    y = int(y)

    # 防止文字超出画面
    if x + text_w + 8 > w:
        x = max(0, w - text_w - 8)
    if y - text_h - 8 < 0:
        y = text_h + 8
    if y + baseline + 4 > h:
        y = h - baseline - 4

    cv2.rectangle(
        image,
        (x - 3, y - text_h - 4),
        (x + text_w + 3, y + baseline + 3),
        bg_color,
        -1
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness
    )


def draw_3d_coords_near_keypoints(image, all_persons_3d):
    """
    将每个关键点的3D坐标直接画在对应点旁边。
    """
    if len(all_persons_3d) == 0:
        draw_text_with_background(
            image,
            "No person detected",
            20,
            35,
            font_scale=0.7,
            text_color=(0, 255, 255),
            thickness=2
        )
        return image

    # 每个人用不同的文字偏移，尽量减少多人时文字重叠
    person_offsets = [
        (8, -8),
        (8, 18),
        (-180, -8),
        (-180, 18),
    ]

    for person in all_persons_3d:
        person_id = person["id"]
        keypoints = person["keypoints"]

        offset_x, offset_y = person_offsets[person_id % len(person_offsets)]

        for name, value in keypoints.items():
            if value is None:
                continue

            x_proc, y_proc = value["pixel_proc"]
            X, Y, Z = value["xyz_mm"]

            short_name = DISPLAY_NAMES.get(name, name)

            # 文字内容：关节点名 + 三维坐标
            # 为了不太挤，这里取整数mm
            text = f"{short_name}:({X:.0f},{Y:.0f},{Z:.0f})"

            px = int(round(x_proc))
            py = int(round(y_proc))

            # 在点旁边再画一个小圆点，强调这是有3D坐标的点
            cv2.circle(image, (px, py), 4, (0, 255, 255), -1)

            draw_text_with_background(
                image,
                text,
                px + offset_x,
                py + offset_y,
                font_scale=0.36,
                text_color=(0, 255, 255),
                thickness=1
            )

    return image


# ============================================================
# 5. 主函数
# ============================================================
def main():
    # =========================
    # 参数区
    # =========================
    MODEL_PATH = "yolov8n-pose.pt"

    # YOLO推理和显示尺寸
    # 如果电脑卡，可以改成 640, 360
    PROC_W, PROC_H = 640, 360

    KEYPOINT_CONF_THRES = 0.30
    DEPTH_WINDOW = 5

    # 是否显示所有17个关键点的坐标
    # True：显示17个点
    # False：只显示肩、肘、腕、髋、膝、踝
    SHOW_ALL_KEYPOINTS = True

    IMPORTANT_KEYPOINTS = {
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    }

    # =========================
    # 加载 YOLO Pose 模型
    # =========================
    print(">>> 正在加载 YOLOv8 Pose 模型...")
    pose_model = YOLO(MODEL_PATH)
    print(">>> YOLO模型加载完成")

    # =========================
    # 初始化 Orbbec 相机
    # =========================
    pipeline = Pipeline()
    config = Config()

    try:
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

        try:
            color_profile = color_profile_list.get_video_stream_profile(
                1920,
                1080,
                OBFormat.RGB888,
                30
            )
        except Exception:
            color_profile = color_profile_list.get_default_video_stream_profile()

        config.enable_stream(color_profile)

        # 深度对齐到彩色图
        config.set_align_mode(OBAlignMode.HW_MODE)

    except Exception as e:
        print(f"❌ 配置相机流失败: {e}")
        return

    try:
        print(">>> 正在启动 Orbbec 相机...")
        pipeline.start(config)
        print(">>> 相机启动成功，已开启硬件D2C对齐")

        try:
            pipeline.enable_frame_sync()
            print(">>> 已开启 RGB 与 Depth 帧同步")
        except Exception as e:
            print(f"⚠️ 帧同步开启失败，但程序继续运行: {e}")

        # =========================
        # 获取 RGB 相机内参
        # =========================
        camera_param = pipeline.get_camera_param()
        rgb_intrinsic = camera_param.rgb_intrinsic

        fx = rgb_intrinsic.fx
        fy = rgb_intrinsic.fy
        cx = rgb_intrinsic.cx
        cy = rgb_intrinsic.cy

        print("\n>>> RGB相机内参：")
        print(f"fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
        print(">>> 开始实时显示3D骨骼点，按 q 或 ESC 退出\n")

        while True:
            frames = pipeline.wait_for_frames(1000)

            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            # =========================
            # 读取 RGB 图像
            # =========================
            try:
                color_image = decode_color_frame(color_frame)
            except Exception as e:
                print(f"⚠️ 彩色图像解码失败: {e}")
                continue

            color_h, color_w = color_image.shape[:2]

            # =========================
            # 读取 Depth 图像
            # =========================
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_h = depth_frame.get_height()
            depth_w = depth_frame.get_width()

            try:
                depth_matrix = depth_data.reshape((depth_h, depth_w))
            except Exception as e:
                print(f"⚠️ 深度图像reshape失败: {e}")
                continue

            try:
                depth_scale = depth_frame.get_depth_scale()
            except Exception:
                depth_scale = 1.0

            # =========================
            # 缩放 RGB 图像，用于 YOLO 推理和实时显示
            # =========================
            proc_color = cv2.resize(color_image, (PROC_W, PROC_H))

            # =========================
            # YOLO Pose 推理
            # =========================
            results = pose_model(proc_color, verbose=False)
            result = results[0]

            # YOLO自带骨架绘制
            annotated_rgb = result.plot()

            # =========================
            # 生成对齐深度图显示
            # =========================
            proc_depth_matrix = cv2.resize(
                depth_matrix,
                (PROC_W, PROC_H),
                interpolation=cv2.INTER_NEAREST
            )

            depth_display = cv2.normalize(
                proc_depth_matrix,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )

            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            annotated_depth = result.plot(img=depth_colormap.copy())

            # =========================
            # 计算所有人的3D骨骼点
            # =========================
            all_persons_3d = []

            if result.keypoints is not None and result.keypoints.xy is not None:
                keypoints_xy = result.keypoints.xy.cpu().numpy()

                if result.keypoints.conf is not None:
                    keypoints_conf = result.keypoints.conf.cpu().numpy()
                else:
                    keypoints_conf = np.ones(
                        (len(keypoints_xy), len(COCO_KEYPOINT_NAMES)),
                        dtype=np.float32
                    )

                boxes_xyxy = None
                if result.boxes is not None and result.boxes.xyxy is not None:
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()

                for person_id, kp_xy_proc in enumerate(keypoints_xy):
                    person_3d = {
                        "id": person_id,
                        "keypoints": {}
                    }

                    # 给每个人画ID和检测框
                    if boxes_xyxy is not None and person_id < len(boxes_xyxy):
                        x1, y1, x2, y2 = boxes_xyxy[person_id].astype(int)

                        cv2.rectangle(
                            annotated_rgb,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            annotated_rgb,
                            f"ID:{person_id}",
                            (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                    for kpt_id, name in enumerate(COCO_KEYPOINT_NAMES):
                        # 如果只想显示主要关节点，就跳过眼睛耳朵鼻子
                        if not SHOW_ALL_KEYPOINTS and name not in IMPORTANT_KEYPOINTS:
                            continue

                        x_proc, y_proc = kp_xy_proc[kpt_id]
                        conf = float(keypoints_conf[person_id][kpt_id])

                        if conf < KEYPOINT_CONF_THRES:
                            person_3d["keypoints"][name] = None
                            continue

                        if x_proc <= 0 or y_proc <= 0:
                            person_3d["keypoints"][name] = None
                            continue

                        # YOLO是在 proc_color 上识别的，需要映射回原始RGB分辨率
                        u_rgb = x_proc * color_w / PROC_W
                        v_rgb = y_proc * color_h / PROC_H

                        # 如果D2C后深度图尺寸和RGB尺寸仍不完全一致，就再做比例映射
                        u_depth = u_rgb * depth_w / color_w
                        v_depth = v_rgb * depth_h / color_h

                        depth_mm = get_valid_depth_mm(
                            depth_matrix,
                            u_depth,
                            v_depth,
                            depth_scale=depth_scale,
                            window_size=DEPTH_WINDOW
                        )

                        if depth_mm is None:
                            person_3d["keypoints"][name] = None
                            continue

                        X, Y, Z = pixel_to_camera_3d(
                            u_rgb,
                            v_rgb,
                            depth_mm,
                            fx,
                            fy,
                            cx,
                            cy
                        )

                        person_3d["keypoints"][name] = {
                            "pixel_proc": (float(x_proc), float(y_proc)),
                            "pixel_rgb": (float(u_rgb), float(v_rgb)),
                            "xyz_mm": (X, Y, Z),
                            "confidence": conf
                        }

                    all_persons_3d.append(person_3d)

            # =========================
            # 关键：把3D坐标画在对应关键点旁边
            # =========================
            annotated_rgb = draw_3d_coords_near_keypoints(
                annotated_rgb,
                all_persons_3d
            )

            # =========================
            # 实时显示
            # =========================
            cv2.imshow("YOLO Pose RGB + 3D Keypoints", annotated_rgb)
            cv2.imshow("YOLO Pose Depth Aligned", annotated_depth)

            key = cv2.waitKey(1)

            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n>>> 用户手动中断程序")

    except Exception as e:
        print(f"❌ 程序运行错误: {e}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(">>> 相机已关闭，窗口已销毁")


if __name__ == "__main__":
    main()
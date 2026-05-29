# -*- coding: utf-8 -*-
"""
PyQt + Orbbec + YOLOv8-Pose 康复视觉评定界面
功能：
1. PyQt 主界面：左侧实时视频，右侧曲线卡片，底部开始/结束测量。
2. 新增长度测量：左/右侧上臂、前臂、大腿、小腿 3D 长度实时曲线与数值。
3. 多线程加速：相机采集线程 + YOLO/3D计算线程 + GUI主线程；队列只保留最新帧，降低延迟、提升显示帧率。

依赖：
    pip install PyQt5 ultralytics opencv-python numpy
    # 另需安装并配置 pyorbbecsdk

退出：点击“结束测量”或关闭窗口。
"""

import sys
import time
import queue
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode, OBFormat

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QFrame,
    QSizePolicy,
)


# ============================================================
# 1. 全局参数区
# ============================================================
MODEL_PATH = "yolov8n-pose.pt"

# YOLO推理与显示尺寸。若电脑较卡，可改成 640 x 360。
PROC_W, PROC_H = 800, 450
YOLO_IMGSZ = 640

KEYPOINT_CONF_THRES = 0.30
DEPTH_WINDOW = 5

# 是否在视频关键点旁显示3D坐标；坐标较多时可改为 False。
SHOW_3D_COORDS = True
SHOW_ALL_KEYPOINTS = False

# 历史曲线长度。越大越能看趋势，但绘制负担也会增加。
HISTORY_LEN = 160

# 队列长度为1表示只处理最新帧，避免推理跟不上时越积越慢。
FRAME_QUEUE_SIZE = 1


# ============================================================
# 2. COCO 17个关键点定义
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

SKELETON_PAIRS = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4)
]

IMPORTANT_KEYPOINTS = {
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
}


# ============================================================
# 3. 深度、3D坐标、图像解码
# ============================================================
def get_valid_depth_mm(depth_matrix, u, v, depth_scale=1.0, window_size=5):
    """在关键点附近取小窗口，对非0深度取中值，返回单位mm。"""
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

    return float(np.median(valid) * depth_scale)


def pixel_to_camera_3d(u, v, depth_mm, fx, fy, cx, cy):
    """像素坐标 + 深度 -> 相机坐标系3D坐标，单位mm。"""
    z = depth_mm
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return float(x), float(y), float(z)


def decode_color_frame(color_frame):
    """将 Orbbec color_frame 解码为 OpenCV BGR 图像。"""
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
# 4. OpenCV 绘图辅助
# ============================================================
def draw_text_with_background(
    image,
    text,
    x,
    y,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.42,
    text_color=(0, 255, 255),
    bg_color=(0, 0, 0),
    thickness=1,
):
    h, w = image.shape[:2]
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x = int(x)
    y = int(y)

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
        -1,
    )
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_selected_pose(image, kp_xy_proc, kp_conf, box_xyxy=None, score=None, conf_thres=0.30):
    """只画当前选中的人：ID0。"""
    if kp_xy_proc is None or kp_conf is None:
        return image

    # 骨架线
    for a, b in SKELETON_PAIRS:
        if a >= len(kp_xy_proc) or b >= len(kp_xy_proc):
            continue
        if kp_conf[a] < conf_thres or kp_conf[b] < conf_thres:
            continue
        xa, ya = kp_xy_proc[a]
        xb, yb = kp_xy_proc[b]
        if xa <= 0 or ya <= 0 or xb <= 0 or yb <= 0:
            continue
        cv2.line(image, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), (58, 220, 92), 2)

    # 关键点
    for kpt_id, (x, y) in enumerate(kp_xy_proc):
        if kp_conf[kpt_id] < conf_thres:
            continue
        if x <= 0 or y <= 0:
            continue
        cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)

    # 检测框与ID
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (32, 80, 255), 2)
        label = "ID:0"
        if score is not None:
            label += f"  conf:{score:.2f}"
        draw_text_with_background(
            image,
            label,
            x1,
            max(24, y1 - 8),
            font_scale=0.62,
            text_color=(60, 120, 255),
            thickness=2,
        )
    return image


def draw_3d_coords_near_keypoints(image, person_3d, show_all_keypoints=False, important_keypoints=None):
    """将 ID0 关键点3D坐标画在对应点附近。"""
    if important_keypoints is None:
        important_keypoints = set()

    if person_3d is None or len(person_3d.get("keypoints", {})) == 0:
        draw_text_with_background(
            image,
            "No ID0 person detected",
            18,
            34,
            font_scale=0.68,
            text_color=(0, 255, 255),
            thickness=2,
        )
        return image

    keypoints = person_3d["keypoints"]
    offset_map = {
        "left_shoulder": (8, -10),
        "right_shoulder": (-170, -10),
        "left_elbow": (8, 0),
        "right_elbow": (-170, 0),
        "left_wrist": (8, 12),
        "right_wrist": (-170, 12),
        "left_hip": (8, -10),
        "right_hip": (-170, -10),
        "left_knee": (8, 0),
        "right_knee": (-170, 0),
        "left_ankle": (8, 12),
        "right_ankle": (-170, 12),
    }

    for name, value in keypoints.items():
        if not show_all_keypoints and name not in important_keypoints:
            continue
        if value is None:
            continue
        x_proc, y_proc = value["pixel_proc"]
        X, Y, Z = value["xyz_mm"]
        px = int(round(x_proc))
        py = int(round(y_proc))
        short_name = DISPLAY_NAMES.get(name, name)
        text = f"{short_name}:({X:.0f},{Y:.0f},{Z:.0f})"
        cv2.circle(image, (px, py), 4, (0, 255, 255), -1)
        ox, oy = offset_map.get(name, (8, -8))
        draw_text_with_background(
            image,
            text,
            px + ox,
            py + oy,
            font_scale=0.34,
            text_color=(0, 255, 255),
            thickness=1,
        )
    return image


def draw_video_status_bar(image, fps=None, score=None):
    """视频左上角状态条。"""
    text = "Measuring | ID0 = highest confidence person"
    if score is not None:
        text += f" | conf:{score:.2f}"
    if fps is not None:
        text += f" | FPS:{fps:.1f}"

    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (min(image.shape[1] - 8, 520), 42), (16, 18, 22), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, image)
    cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 520), 42), (90, 100, 115), 1)
    cv2.putText(image, text, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 245, 255), 1, cv2.LINE_AA)
    return image


# ============================================================
# 5. 运动学计算：长度、角度、速度
# ============================================================
def get_xyz(person_3d, name):
    if person_3d is None:
        return None
    value = person_3d.get("keypoints", {}).get(name)
    if value is None:
        return None
    xyz = value.get("xyz_mm")
    if xyz is None:
        return None
    return np.array(xyz, dtype=np.float32)


def distance_3d(p1, p2):
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(p1 - p2))


def angle_3d(p_a, p_b, p_c):
    """计算三点夹角 ∠ABC，p_b 为关节中心点，单位degree。"""
    if p_a is None or p_b is None or p_c is None:
        return None
    v1 = p_a - p_b
    v2 = p_c - p_b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def compute_metrics(person_3d, prev_state, curr_time):
    """
    输出 metrics：
    lengths: 肩-肘、肘-腕、髋-膝、膝-踝，单位mm
    angles: 肩、肘、膝，单位deg
    linear_velocity: 手腕、脚踝，单位mm/s
    angular_velocity: 肩、肘、膝角速度，单位deg/s
    """
    points = {name: get_xyz(person_3d, name) for name in COCO_KEYPOINT_NAMES}

    lengths = {
        "left_upper_arm": distance_3d(points["left_shoulder"], points["left_elbow"]),
        "right_upper_arm": distance_3d(points["right_shoulder"], points["right_elbow"]),
        "left_forearm": distance_3d(points["left_elbow"], points["left_wrist"]),
        "right_forearm": distance_3d(points["right_elbow"], points["right_wrist"]),
        "left_thigh": distance_3d(points["left_hip"], points["left_knee"]),
        "right_thigh": distance_3d(points["right_hip"], points["right_knee"]),
        "left_shank": distance_3d(points["left_knee"], points["left_ankle"]),
        "right_shank": distance_3d(points["right_knee"], points["right_ankle"]),
    }

    angles = {
        "left_shoulder": angle_3d(points["left_elbow"], points["left_shoulder"], points["left_hip"]),
        "right_shoulder": angle_3d(points["right_elbow"], points["right_shoulder"], points["right_hip"]),
        "left_elbow": angle_3d(points["left_shoulder"], points["left_elbow"], points["left_wrist"]),
        "right_elbow": angle_3d(points["right_shoulder"], points["right_elbow"], points["right_wrist"]),
        "left_knee": angle_3d(points["left_hip"], points["left_knee"], points["left_ankle"]),
        "right_knee": angle_3d(points["right_hip"], points["right_knee"], points["right_ankle"]),
    }

    linear_velocity = {
        "left_wrist": None,
        "right_wrist": None,
        "left_ankle": None,
        "right_ankle": None,
    }
    angular_velocity = {
        "left_shoulder": None,
        "right_shoulder": None,
        "left_elbow": None,
        "right_elbow": None,
        "left_knee": None,
        "right_knee": None,
    }

    prev_time = prev_state.get("time")
    prev_points = prev_state.get("points", {})
    prev_angles = prev_state.get("angles", {})

    if prev_time is not None:
        dt = curr_time - prev_time
        if 1e-3 < dt < 1.0:
            for name in linear_velocity.keys():
                p_now = points.get(name)
                p_prev = prev_points.get(name)
                if p_now is not None and p_prev is not None:
                    linear_velocity[name] = float(np.linalg.norm(p_now - p_prev) / dt)

            for name in angular_velocity.keys():
                a_now = angles.get(name)
                a_prev = prev_angles.get(name)
                if a_now is not None and a_prev is not None:
                    angular_velocity[name] = float(abs(a_now - a_prev) / dt)

    prev_state["time"] = curr_time
    prev_state["points"] = points
    prev_state["angles"] = angles

    return {
        "lengths": lengths,
        "angles": angles,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
    }


def select_id0_by_highest_confidence(result, keypoints_xy, keypoints_conf):
    """当前帧只选择一个人作为 ID0：优先检测框置信度最高，否则关键点平均置信度最高。"""
    if keypoints_xy is None or len(keypoints_xy) == 0:
        return None, None, None

    boxes_xyxy = None
    scores = None

    if result.boxes is not None:
        if result.boxes.xyxy is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        if result.boxes.conf is not None:
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)

    if scores is None or len(scores) != len(keypoints_xy):
        scores = np.zeros((len(keypoints_xy),), dtype=np.float32)
        for i in range(len(keypoints_xy)):
            conf_i = keypoints_conf[i]
            valid = conf_i[conf_i > 0]
            scores[i] = float(np.mean(valid)) if valid.size > 0 else 0.0

    selected_index = int(np.argmax(scores))
    selected_score = float(scores[selected_index])
    return selected_index, selected_score, boxes_xyxy


# ============================================================
# 6. 多线程数据结构与线程
# ============================================================
@dataclass
class FrameBundle:
    color_image: np.ndarray
    depth_matrix: np.ndarray
    depth_scale: float
    fx: float
    fy: float
    cx: float
    cy: float


class CameraThread(QThread):
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, frame_queue, parent=None):
        super().__init__(parent)
        self.frame_queue = frame_queue
        self._running = False
        self.pipeline = None

    def stop(self):
        self._running = False

    def _put_latest(self, item):
        """队列只保留最新帧；满了就丢掉旧帧。"""
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(item)
        except queue.Full:
            pass

    def run(self):
        self._running = True
        self.pipeline = Pipeline()
        config = Config()

        try:
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)

            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile = color_profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB888, 30)
            except Exception:
                color_profile = color_profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)

            # 深度对齐到彩色图。
            config.set_align_mode(OBAlignMode.HW_MODE)
        except Exception as e:
            self.error_signal.emit(f"配置相机流失败：{e}")
            return

        try:
            self.status_signal.emit("正在启动 Orbbec 相机...")
            self.pipeline.start(config)
            self.status_signal.emit("相机启动成功，已开启硬件D2C对齐")

            try:
                self.pipeline.enable_frame_sync()
                self.status_signal.emit("已开启 RGB 与 Depth 帧同步")
            except Exception as e:
                self.status_signal.emit(f"帧同步开启失败，继续运行：{e}")

            camera_param = self.pipeline.get_camera_param()
            rgb_intrinsic = camera_param.rgb_intrinsic
            fx, fy, cx, cy = rgb_intrinsic.fx, rgb_intrinsic.fy, rgb_intrinsic.cx, rgb_intrinsic.cy
            self.status_signal.emit(f"RGB内参：fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

            while self._running:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    continue

                try:
                    color_image = decode_color_frame(color_frame)
                except Exception as e:
                    self.status_signal.emit(f"彩色图像解码失败：{e}")
                    continue

                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_h = depth_frame.get_height()
                depth_w = depth_frame.get_width()
                try:
                    depth_matrix = depth_data.reshape((depth_h, depth_w))
                except Exception as e:
                    self.status_signal.emit(f"深度图像reshape失败：{e}")
                    continue

                try:
                    depth_scale = depth_frame.get_depth_scale()
                except Exception:
                    depth_scale = 1.0

                self._put_latest(
                    FrameBundle(
                        color_image=color_image,
                        depth_matrix=depth_matrix,
                        depth_scale=depth_scale,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                    )
                )

        except Exception as e:
            self.error_signal.emit(f"相机线程错误：{e}")

        finally:
            try:
                if self.pipeline is not None:
                    self.pipeline.stop()
            except Exception:
                pass
            self.status_signal.emit("相机已关闭")


class PoseProcessThread(QThread):
    frame_signal = pyqtSignal(object)   # dict: image, metrics, fps, score
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, frame_queue, parent=None):
        super().__init__(parent)
        self.frame_queue = frame_queue
        self._running = False
        self.pose_model = None
        self.prev_state = {"time": None, "points": {}, "angles": {}}
        self.fps_value = None
        self.last_time = None

    def stop(self):
        self._running = False

    def _reset_velocity_state(self):
        self.prev_state["time"] = None
        self.prev_state["points"] = {}
        self.prev_state["angles"] = {}

    def _load_model(self):
        self.status_signal.emit("正在加载 YOLOv8 Pose 模型...")
        self.pose_model = YOLO(MODEL_PATH)

        # 自动选择CUDA/CPU。未安装torch或无GPU时会回退CPU。
        try:
            import torch
            if torch.cuda.is_available():
                self.device = 0
                self.status_signal.emit("YOLO模型加载完成，当前使用 CUDA")
            else:
                self.device = "cpu"
                self.status_signal.emit("YOLO模型加载完成，当前使用 CPU")
        except Exception:
            self.device = "cpu"
            self.status_signal.emit("YOLO模型加载完成，当前使用 CPU")

    def run(self):
        self._running = True
        try:
            self._load_model()
        except Exception as e:
            self.error_signal.emit(f"YOLO模型加载失败：{e}")
            return

        while self._running:
            try:
                bundle = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                output = self.process_one_frame(bundle)
                if output is not None:
                    self.frame_signal.emit(output)
            except Exception as e:
                self.error_signal.emit(f"姿态/计算线程错误：{e}")
                self._reset_velocity_state()

    def process_one_frame(self, bundle: FrameBundle):
        color_image = bundle.color_image
        depth_matrix = bundle.depth_matrix
        depth_scale = bundle.depth_scale
        fx, fy, cx, cy = bundle.fx, bundle.fy, bundle.cx, bundle.cy

        color_h, color_w = color_image.shape[:2]
        depth_h, depth_w = depth_matrix.shape[:2]

        proc_color = cv2.resize(color_image, (PROC_W, PROC_H))

        tic = time.perf_counter()
        results = self.pose_model(
            proc_color,
            verbose=False,
            imgsz=YOLO_IMGSZ,
            device=self.device,
        )
        result = results[0]
        annotated_rgb = proc_color.copy()

        person_3d = None
        metrics = None
        selected_score = None

        if result.keypoints is not None and result.keypoints.xy is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()

            if result.keypoints.conf is not None:
                keypoints_conf = result.keypoints.conf.cpu().numpy()
            else:
                keypoints_conf = np.ones((len(keypoints_xy), len(COCO_KEYPOINT_NAMES)), dtype=np.float32)

            selected_index, selected_score, boxes_xyxy = select_id0_by_highest_confidence(
                result,
                keypoints_xy,
                keypoints_conf,
            )

            if selected_index is not None:
                kp_xy_proc = keypoints_xy[selected_index]
                kp_conf = keypoints_conf[selected_index]

                selected_box = None
                if boxes_xyxy is not None and selected_index < len(boxes_xyxy):
                    selected_box = boxes_xyxy[selected_index]

                annotated_rgb = draw_selected_pose(
                    annotated_rgb,
                    kp_xy_proc,
                    kp_conf,
                    box_xyxy=selected_box,
                    score=selected_score,
                    conf_thres=KEYPOINT_CONF_THRES,
                )

                person_3d = {"id": 0, "keypoints": {}}

                for kpt_id, name in enumerate(COCO_KEYPOINT_NAMES):
                    x_proc, y_proc = kp_xy_proc[kpt_id]
                    conf = float(kp_conf[kpt_id])

                    if conf < KEYPOINT_CONF_THRES or x_proc <= 0 or y_proc <= 0:
                        person_3d["keypoints"][name] = None
                        continue

                    # YOLO 在缩放后的 proc_color 上识别，需要映射回原始RGB分辨率。
                    u_rgb = x_proc * color_w / PROC_W
                    v_rgb = y_proc * color_h / PROC_H

                    # 如果D2C后深度图尺寸和RGB尺寸仍不完全一致，再做比例映射。
                    u_depth = u_rgb * depth_w / color_w
                    v_depth = v_rgb * depth_h / color_h

                    depth_mm = get_valid_depth_mm(
                        depth_matrix,
                        u_depth,
                        v_depth,
                        depth_scale=depth_scale,
                        window_size=DEPTH_WINDOW,
                    )

                    if depth_mm is None:
                        person_3d["keypoints"][name] = None
                        continue

                    X, Y, Z = pixel_to_camera_3d(u_rgb, v_rgb, depth_mm, fx, fy, cx, cy)
                    person_3d["keypoints"][name] = {
                        "pixel_proc": (float(x_proc), float(y_proc)),
                        "pixel_rgb": (float(u_rgb), float(v_rgb)),
                        "xyz_mm": (X, Y, Z),
                        "confidence": conf,
                    }

                curr_time = time.time()
                metrics = compute_metrics(person_3d, self.prev_state, curr_time)

        if person_3d is None:
            self._reset_velocity_state()

        if SHOW_3D_COORDS:
            annotated_rgb = draw_3d_coords_near_keypoints(
                annotated_rgb,
                person_3d,
                show_all_keypoints=SHOW_ALL_KEYPOINTS,
                important_keypoints=IMPORTANT_KEYPOINTS,
            )

        # 统计处理帧率：包括 YOLO 推理、3D坐标、绘制。
        toc = time.perf_counter()
        dt = toc - (self.last_time if self.last_time is not None else tic)
        self.last_time = toc
        if dt > 1e-6:
            inst_fps = 1.0 / dt
            self.fps_value = inst_fps if self.fps_value is None else (0.90 * self.fps_value + 0.10 * inst_fps)

        annotated_rgb = draw_video_status_bar(annotated_rgb, fps=self.fps_value, score=selected_score)

        return {
            "image_bgr": annotated_rgb,
            "metrics": metrics,
            "fps": self.fps_value,
            "score": selected_score,
        }


# ============================================================
# 7. PyQt 自绘曲线卡片
# ============================================================
def safe_float(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
        if not np.isfinite(value):
            return np.nan
        return value
    except Exception:
        return np.nan


def fmt_value(value, digits=0, unit=""):
    try:
        value = float(value)
        if not np.isfinite(value):
            return "--"
        return f"{value:.{digits}f}{unit}"
    except Exception:
        return "--"


class PlotCard(QFrame):
    """纯 PyQt 自绘实时曲线卡片，避免额外依赖 pyqtgraph。"""

    def __init__(self, title, unit, series, fixed_ylim=None, force_zero_min=True, parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.series = series  # [(key, label, QColor), ...]
        self.fixed_ylim = fixed_ylim
        self.force_zero_min = force_zero_min
        self.max_len = HISTORY_LEN
        self.history = {key: deque(maxlen=self.max_len) for key, _, _ in series}
        self.latest = {key: np.nan for key, _, _ in series}

        self.setMinimumSize(QSize(250, 145))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

    def clear(self):
        for key in self.history:
            self.history[key].clear()
            self.latest[key] = np.nan
        self.update()

    def append_values(self, values):
        values = values or {}
        for key, _, _ in self.series:
            v = safe_float(values.get(key))
            self.history[key].append(v)
            self.latest[key] = v
        self.update()

    def _finite_values(self):
        values = []
        for dq in self.history.values():
            values.extend([float(v) for v in dq if np.isfinite(v)])
        return values

    def _get_ylim(self):
        if self.fixed_ylim is not None:
            return self.fixed_ylim
        vals = self._finite_values()
        if len(vals) == 0:
            return 0.0, 1.0
        y_min = min(vals)
        y_max = max(vals)
        if self.force_zero_min:
            y_min = 0.0
        if abs(y_max - y_min) < 1e-6:
            y_max = y_min + 1.0
        margin = 0.12 * (y_max - y_min)
        y_max += margin
        if not self.force_zero_min:
            y_min -= margin
        return y_min, y_max

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(4, 4, -4, -4)
        bg = QColor("#eaf7ef")
        border = QColor("#93b9a0")
        title_color = QColor("#143b2a")
        grid_color = QColor("#c9dfd1")
        axis_color = QColor("#7b9282")
        text_dim = QColor("#5c6d62")

        painter.setPen(QPen(border, 1))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(rect, 10, 10)

        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        painter.setPen(title_color)
        painter.drawText(rect.left() + 12, rect.top() + 23, self.title)

        painter.setFont(QFont("Arial", 8))
        painter.setPen(text_dim)
        painter.drawText(rect.right() - 58, rect.top() + 22, self.unit)

        plot_left = rect.left() + 44
        plot_top = rect.top() + 38
        plot_right = rect.right() - 10
        plot_bottom = rect.bottom() - 18
        plot_w = max(1, plot_right - plot_left)
        plot_h = max(1, plot_bottom - plot_top)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#f8fffa")))
        painter.drawRoundedRect(plot_left, plot_top, plot_w, plot_h, 6, 6)

        y_min, y_max = self._get_ylim()
        y_span = max(1e-6, y_max - y_min)

        # 网格和Y轴文字
        painter.setFont(QFont("Arial", 7))
        for i in range(4):
            yy = plot_top + int(i * plot_h / 3)
            val = y_max - i * y_span / 3
            painter.setPen(QPen(grid_color, 1))
            painter.drawLine(plot_left, yy, plot_right, yy)
            painter.setPen(text_dim)
            painter.drawText(rect.left() + 8, yy + 4, f"{val:.0f}")

        painter.setPen(QPen(axis_color, 1))
        painter.drawLine(plot_left, plot_top, plot_left, plot_bottom)
        painter.drawLine(plot_left, plot_bottom, plot_right, plot_bottom)

        # 曲线
        max_len_now = 1
        for dq in self.history.values():
            max_len_now = max(max_len_now, len(dq))

        if max_len_now >= 2:
            for key, _, color in self.series:
                dq = list(self.history.get(key, []))
                prev = None
                painter.setPen(QPen(color, 2))
                for i, v in enumerate(dq):
                    if not np.isfinite(v):
                        prev = None
                        continue
                    xx = plot_left + int(i * (plot_w - 1) / max(1, max_len_now - 1))
                    yy = plot_bottom - int((float(v) - y_min) / y_span * plot_h)
                    yy = max(plot_top, min(plot_bottom, yy))
                    curr = (xx, yy)
                    if prev is not None:
                        painter.drawLine(prev[0], prev[1], curr[0], curr[1])
                    prev = curr
        else:
            painter.setPen(text_dim)
            painter.setFont(QFont("Microsoft YaHei", 9))
            painter.drawText(plot_left + 10, plot_top + 28, "等待数据...")

        # 图例与当前数值：压在右上角，避免窗口横向过长。
        legend_x = plot_right - 132
        legend_y = plot_top + 6
        legend_w = 126
        legend_h = 16 * len(self.series) + 8
        painter.setPen(QPen(QColor("#d6eadc"), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255, 225)))
        painter.drawRoundedRect(legend_x, legend_y, legend_w, legend_h, 6, 6)

        painter.setFont(QFont("Arial", 7))
        for idx, (key, label, color) in enumerate(self.series):
            yy = legend_y + 15 + idx * 16
            painter.setPen(QPen(color, 2))
            painter.drawLine(legend_x + 6, yy - 4, legend_x + 20, yy - 4)
            painter.setPen(color)
            painter.drawText(legend_x + 25, yy, f"{label}:{fmt_value(self.latest.get(key), 0)}")

        painter.end()


# ============================================================
# 8. 主界面
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("康复视觉评定系统 - PyQt多线程版")
        self.resize(1420, 860)

        self.frame_queue = None
        self.camera_thread = None
        self.pose_thread = None
        self.running = False

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 14, 18, 14)
        root_layout.setSpacing(12)

        header = QLabel("康复视觉评定可视化界面")
        header.setObjectName("HeaderLabel")
        root_layout.addWidget(header)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)
        root_layout.addLayout(content_layout, stretch=1)

        # 左侧视频区域
        left_panel = QFrame()
        left_panel.setObjectName("MainPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(12)

        self.video_label = QLabel("点击“开始测量”后显示实时视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setMinimumSize(760, 430)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_label, stretch=1)

        control_layout = QHBoxLayout()
        control_layout.addStretch(1)
        self.start_btn = QPushButton("开始测量")
        self.stop_btn = QPushButton("结束测量")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_measurement)
        self.stop_btn.clicked.connect(self.stop_measurement)
        control_layout.addWidget(self.start_btn)
        control_layout.addSpacing(22)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch(1)
        left_layout.addLayout(control_layout)

        self.status_label = QLabel("状态：未开始")
        self.status_label.setObjectName("StatusLabel")
        left_layout.addWidget(self.status_label)

        content_layout.addWidget(left_panel, stretch=3)

        # 右侧曲线区域：原6类曲线 + 新增左右长度测量。
        right_panel = QFrame()
        right_panel.setObjectName("MainPanel")
        right_layout = QGridLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setHorizontalSpacing(10)
        right_layout.setVerticalSpacing(10)

        self.cards = {}
        self._create_plot_cards()

        # 2列 x 4行：右/左 关节角度、线速度、角速度、长度测量
        right_layout.addWidget(self.cards["right_angles"], 0, 0)
        right_layout.addWidget(self.cards["left_angles"], 0, 1)
        right_layout.addWidget(self.cards["right_linear_velocity"], 1, 0)
        right_layout.addWidget(self.cards["left_linear_velocity"], 1, 1)
        right_layout.addWidget(self.cards["right_angular_velocity"], 2, 0)
        right_layout.addWidget(self.cards["left_angular_velocity"], 2, 1)
        right_layout.addWidget(self.cards["right_lengths"], 3, 0)
        right_layout.addWidget(self.cards["left_lengths"], 3, 1)

        content_layout.addWidget(right_panel, stretch=2)

    def _create_plot_cards(self):
        blue = QColor("#2f80ed")
        green = QColor("#27ae60")
        orange = QColor("#f2994a")
        red = QColor("#eb5757")
        purple = QColor("#9b51e0")
        cyan = QColor("#00a6a6")

        self.cards["right_angles"] = PlotCard(
            "右关节角度曲线",
            "deg",
            [
                ("right_shoulder", "肩", blue),
                ("right_elbow", "肘", green),
                ("right_knee", "膝", orange),
            ],
            fixed_ylim=(0.0, 180.0),
        )
        self.cards["left_angles"] = PlotCard(
            "左关节角度曲线",
            "deg",
            [
                ("left_shoulder", "肩", blue),
                ("left_elbow", "肘", green),
                ("left_knee", "膝", orange),
            ],
            fixed_ylim=(0.0, 180.0),
        )
        self.cards["right_linear_velocity"] = PlotCard(
            "右线速度曲线",
            "mm/s",
            [
                ("right_wrist", "腕", purple),
                ("right_ankle", "踝", cyan),
            ],
            fixed_ylim=None,
        )
        self.cards["left_linear_velocity"] = PlotCard(
            "左线速度曲线",
            "mm/s",
            [
                ("left_wrist", "腕", purple),
                ("left_ankle", "踝", cyan),
            ],
            fixed_ylim=None,
        )
        self.cards["right_angular_velocity"] = PlotCard(
            "右角速度曲线",
            "deg/s",
            [
                ("right_shoulder", "肩", blue),
                ("right_elbow", "肘", green),
                ("right_knee", "膝", orange),
            ],
            fixed_ylim=None,
        )
        self.cards["left_angular_velocity"] = PlotCard(
            "左角速度曲线",
            "deg/s",
            [
                ("left_shoulder", "肩", blue),
                ("left_elbow", "肘", green),
                ("left_knee", "膝", orange),
            ],
            fixed_ylim=None,
        )
        self.cards["right_lengths"] = PlotCard(
            "右肢段长度测量",
            "mm",
            [
                ("right_upper_arm", "上臂", blue),
                ("right_forearm", "前臂", green),
                ("right_thigh", "大腿", orange),
                ("right_shank", "小腿", red),
            ],
            fixed_ylim=None,
        )
        self.cards["left_lengths"] = PlotCard(
            "左肢段长度测量",
            "mm",
            [
                ("left_upper_arm", "上臂", blue),
                ("left_forearm", "前臂", green),
                ("left_thigh", "大腿", orange),
                ("left_shank", "小腿", red),
            ],
            fixed_ylim=None,
        )

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f5f7fb;
            }
            QLabel#HeaderLabel {
                color: #1546a0;
                font-family: "Microsoft YaHei";
                font-size: 24px;
                font-weight: 700;
            }
            QFrame#MainPanel {
                background: #eef3fb;
                border: 1px solid #b8c7da;
                border-radius: 10px;
            }
            QLabel#VideoLabel {
                background: #151923;
                color: #cfd8e3;
                border: 2px solid #242c3d;
                border-radius: 4px;
                font-family: "Microsoft YaHei";
                font-size: 16px;
            }
            QLabel#StatusLabel {
                color: #405066;
                font-family: "Microsoft YaHei";
                font-size: 13px;
            }
            QPushButton {
                min-width: 108px;
                min-height: 34px;
                border-radius: 8px;
                border: 1px solid #8d7dbd;
                background: #eee7ff;
                color: #21153a;
                font-family: "Microsoft YaHei";
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #e0d3ff;
            }
            QPushButton:disabled {
                background: #e7e9ef;
                color: #9aa3b1;
                border: 1px solid #c4c8d2;
            }
            """
        )

    def clear_all_cards(self):
        for card in self.cards.values():
            card.clear()

    def start_measurement(self):
        if self.running:
            return

        self.clear_all_cards()
        self.video_label.setText("正在启动相机与模型，请稍等...")
        self.status_label.setText("状态：正在启动")

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.camera_thread = CameraThread(self.frame_queue)
        self.pose_thread = PoseProcessThread(self.frame_queue)

        self.camera_thread.status_signal.connect(self.on_status)
        self.camera_thread.error_signal.connect(self.on_error)
        self.pose_thread.status_signal.connect(self.on_status)
        self.pose_thread.error_signal.connect(self.on_error)
        self.pose_thread.frame_signal.connect(self.on_processed_frame)

        # 先开推理线程，模型加载期间相机线程也会持续刷新最新帧，不会阻塞GUI。
        self.pose_thread.start()
        self.camera_thread.start()

        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_measurement(self):
        if not self.running:
            return

        self.status_label.setText("状态：正在结束测量...")
        QApplication.processEvents()

        if self.camera_thread is not None:
            self.camera_thread.stop()
        if self.pose_thread is not None:
            self.pose_thread.stop()

        if self.camera_thread is not None:
            self.camera_thread.wait(1500)
        if self.pose_thread is not None:
            self.pose_thread.wait(1500)

        self.camera_thread = None
        self.pose_thread = None
        self.frame_queue = None
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("状态：测量已结束，曲线保留为最后一次结果")

    def closeEvent(self, event):
        self.stop_measurement()
        event.accept()

    def on_status(self, msg):
        self.status_label.setText(f"状态：{msg}")

    def on_error(self, msg):
        self.status_label.setText(f"错误：{msg}")

    def on_processed_frame(self, payload):
        image_bgr = payload.get("image_bgr")
        metrics = payload.get("metrics")
        fps = payload.get("fps")
        score = payload.get("score")

        self.update_video(image_bgr)
        self.update_cards(metrics)

        score_text = "--" if score is None else f"{score:.2f}"
        fps_text = "--" if fps is None else f"{fps:.1f}"
        self.status_label.setText(f"状态：测量中 | ID0 conf={score_text} | FPS={fps_text}")

    def update_video(self, image_bgr):
        if image_bgr is None:
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def update_cards(self, metrics):
        if metrics is None:
            # 未检测到人时写入 NaN，使曲线自然断开。
            for card in self.cards.values():
                card.append_values({})
            return

        lengths = metrics.get("lengths", {}) or {}
        angles = metrics.get("angles", {}) or {}
        lin_vel = metrics.get("linear_velocity", {}) or {}
        ang_vel = metrics.get("angular_velocity", {}) or {}

        self.cards["right_angles"].append_values(angles)
        self.cards["left_angles"].append_values(angles)
        self.cards["right_linear_velocity"].append_values(lin_vel)
        self.cards["left_linear_velocity"].append_values(lin_vel)
        self.cards["right_angular_velocity"].append_values(ang_vel)
        self.cards["left_angular_velocity"].append_values(ang_vel)
        self.cards["right_lengths"].append_values(lengths)
        self.cards["left_lengths"].append_values(lengths)


# ============================================================
# 9. 入口
# ============================================================
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

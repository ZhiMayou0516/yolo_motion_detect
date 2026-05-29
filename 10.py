# -*- coding: utf-8 -*-
"""
PyQt + Orbbec + YOLOv8-Pose 康复视觉评定界面：优化多线程版

核心优化：
1. 相机线程、姿态推理线程、GUI刷新彻底解耦。
2. 相机线程持续30FPS取最新 RGB+Depth，不等待YOLO。
3. YOLO线程只处理最新帧，旧帧直接丢弃，避免延迟累计。
4. GUI用QTimer刷新：视频约30FPS，曲线约10FPS，状态约2FPS。
5. 视频显示可以接近相机帧率；YOLO姿态结果按最近一次结果异步叠加。
6. 默认关闭关键点旁3D坐标文字，减少OpenCV绘图耗时。

依赖：
    pip install PyQt5 ultralytics opencv-python numpy
    # 另需安装并配置 pyorbbecsdk

运行：
    python pyqt_pose_optimized_multithread_v4_gui_fast.py
"""

import sys
import time
import threading
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode, OBFormat

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QRect
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
# 1. 全局参数区：优先调这里
# ============================================================
MODEL_PATH = "yolov8n-pose.pt"

# 相机请求分辨率。Femto Mega 当前日志显示是 USB2.1，优先用 MJPG 压缩彩色流，避免 RGB888 带宽过高导致没有画面。
REQUEST_COLOR_W = 1280
REQUEST_COLOR_H = 720
REQUEST_COLOR_FPS = 30

# 深度尽量请求30FPS；若启动后拿不到帧，会自动降级到更低带宽组合。
REQUEST_DEPTH_W = 640
REQUEST_DEPTH_H = 576
REQUEST_DEPTH_FPS = 30

# 低带宽模式：优先 MJPG + 30FPS Depth；如果3秒内没有帧，会自动尝试下一组 profile。
LOW_BANDWIDTH_MODE = True
CAMERA_STARTUP_TIMEOUT_SEC = 3.0

# YOLO推理/视频显示统一尺寸。想提速：640x360 + imgsz 512 是比较稳的组合。
PROC_W, PROC_H = 640, 360
YOLO_IMGSZ = 512

# GUI刷新频率。视频30FPS，曲线没必要30FPS，10FPS足够看趋势。
VIDEO_TIMER_MS = 33      # 约30FPS
PLOT_TIMER_MS = 200      # 约5FPS：降低主线程绘图占用，让视频GUI更接近30FPS
STATUS_TIMER_MS = 500    # 约2FPS

KEYPOINT_CONF_THRES = 0.30
YOLO_CONF_THRES = 0.25
MAX_DET = 1              # 单人康复评定建议1；如果要多人候选可改3
DEPTH_WINDOW = 5

# CUDA下启用FP16；CPU下会自动关闭。
ENABLE_CUDA_HALF = True

# 显示控制：视频上画3D坐标文字很耗时，默认关闭。
SHOW_3D_COORDS = False
SHOW_ALL_KEYPOINTS = False
SHOW_BONE_LENGTH_ON_VIDEO = False   # 右侧已有长度曲线；关闭视频骨长浮窗可显著减轻GUI负担
SHOW_SKELETON_ON_VIDEO = True
SHOW_VIDEO_STATUS_BAR = True        # 如果GUI仍低于25，可改False，状态栏底部仍会显示FPS

# 曲线历史长度。越大绘制越重。
HISTORY_LEN = 100

# 没有新姿态结果时，旧骨架最多保留多久。太大会出现骨架滞后明显。
POSE_RESULT_TTL = 0.35


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

# 只有这些点参与3D坐标/长度/角度/速度，少取深度能省一点时间。
METRIC_KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


# ============================================================
# 3. 最新数据缓冲区：只保存最新帧/最新姿态，线程安全
# ============================================================
@dataclass
class FrameBundle:
    frame_id: int
    timestamp: float
    color_image: np.ndarray
    depth_matrix: np.ndarray
    depth_scale: float
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class PoseResultBundle:
    frame_id: int
    timestamp: float
    kp_xy_proc: object = None
    kp_conf: object = None
    box_xyxy: object = None
    score: object = None
    person_3d: object = None
    metrics: object = None
    has_person: bool = False


class LatestBuffer:
    """线程安全最新值缓冲区。set覆盖旧值，get读取最新值。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def set(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value

    def clear(self):
        with self._lock:
            self._value = None


# ============================================================
# 4. 深度、3D坐标、图像解码
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

    patch = depth_matrix[y1:y2, x1:x2]
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


def _fmt_eq(fmt, name):
    """兼容不同版本 pyorbbecsdk 的 OBFormat 枚举名。"""
    return hasattr(OBFormat, name) and fmt == getattr(OBFormat, name)


def decode_color_frame(color_frame):
    """将 Orbbec color_frame 解码为 OpenCV BGR 图像。支持 MJPG/RGB888/BGRA/YUYV/NV12。"""
    color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    h = color_frame.get_height()
    w = color_frame.get_width()
    fmt = color_frame.get_format()

    if _fmt_eq(fmt, "MJPG"):
        color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        if color_image is None:
            raise RuntimeError("MJPG 解码失败")
        return color_image.copy()

    if _fmt_eq(fmt, "RGB888"):
        color_image = color_data.reshape((h, w, 3))
        return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR).copy()

    if _fmt_eq(fmt, "BGRA"):
        color_image = color_data.reshape((h, w, 4))
        return cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR).copy()

    if _fmt_eq(fmt, "YUYV") or _fmt_eq(fmt, "YUY2"):
        color_image = color_data.reshape((h, w, 2))
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUY2).copy()

    if _fmt_eq(fmt, "NV12"):
        color_image = color_data.reshape((h * 3 // 2, w))
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_NV12).copy()

    # 兜底：根据buffer长度判断是不是常见未压缩格式。
    if color_data.size == h * w * 3:
        color_image = color_data.reshape((h, w, 3))
        return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR).copy()
    if color_data.size == h * w * 4:
        color_image = color_data.reshape((h, w, 4))
        return cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR).copy()
    if color_data.size == h * w * 2:
        color_image = color_data.reshape((h, w, 2))
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUY2).copy()

    raise RuntimeError(f"暂不支持的彩色格式: {fmt}, size={color_data.size}, shape=({w},{h})")

def try_get_profile(profile_list, candidates):
    """
    依次尝试多个profile，失败就试下一个，最后fallback到默认profile。
    candidates: [(w,h,format,fps), ...]
    """
    for w, h, fmt, fps in candidates:
        if fmt is None:
            continue
        try:
            return profile_list.get_video_stream_profile(w, h, fmt, fps), f"{w}x{h}@{fps}, {fmt}"
        except Exception:
            continue
    return profile_list.get_default_video_stream_profile(), "default"



def get_profile_or_none(profile_list, w, h, fmt, fps):
    """严格按指定参数取profile；取不到返回None，不自动fallback。"""
    if fmt is None:
        return None
    try:
        return profile_list.get_video_stream_profile(w, h, fmt, fps)
    except Exception:
        return None


def fmt_name(name):
    return getattr(OBFormat, name, None)


# ============================================================
# 5. OpenCV 绘图辅助
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

    for kpt_id, (x, y) in enumerate(kp_xy_proc):
        if kp_conf[kpt_id] < conf_thres:
            continue
        if x <= 0 or y <= 0:
            continue
        cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)

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
            font_scale=0.58,
            text_color=(60, 120, 255),
            thickness=2,
        )
    return image


def draw_3d_coords_near_keypoints(image, person_3d, show_all_keypoints=False, important_keypoints=None):
    """将 ID0 关键点3D坐标画在对应点附近；默认建议关闭以提升FPS。"""
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

    keypoints = person_3d["keypoints"]
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


def fmt_short(value, digits=0, suffix=""):
    if value is None:
        return "--"
    try:
        value = float(value)
        if not np.isfinite(value):
            return "--"
        return f"{value:.{digits}f}{suffix}"
    except Exception:
        return "--"


def put_cv_text(image, text, x, y, scale=0.46, color=(245, 245, 245), thickness=1):
    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_bone_length_on_video(image, metrics, score=None):
    """将骨长参数叠加到 RGB/color 窗口上。"""
    h, w = image.shape[:2]
    lengths = {}
    if metrics is not None:
        lengths = metrics.get("lengths", {}) or {}

    panel_w = min(370, max(300, w - 20))
    panel_h = 145
    x = 10
    y = 10

    # 这里不用半透明 overlay=image.copy()，避免每帧整图复制拖慢GUI。
    cv2.rectangle(image, (x, y), (x + panel_w, y + panel_h), (18, 16, 17), -1)
    cv2.rectangle(image, (x, y), (x + panel_w, y + panel_h), (115, 90, 92), 1)

    title = "Bone Length (3D, mm)"
    if score is not None:
        title += f"   conf:{score:.2f}"
    put_cv_text(image, title, x + 12, y + 24, scale=0.48, color=(245, 245, 245), thickness=1)

    left_x = x + int(panel_w * 0.58)
    right_x = x + int(panel_w * 0.78)
    put_cv_text(image, "Item", x + 14, y + 49, scale=0.36, color=(170, 170, 170))
    put_cv_text(image, "Left", left_x, y + 49, scale=0.36, color=(170, 170, 170))
    put_cv_text(image, "Right", right_x, y + 49, scale=0.36, color=(170, 170, 170))

    rows = [
        ("Upper arm S-E", lengths.get("left_upper_arm"), lengths.get("right_upper_arm")),
        ("Forearm   E-W", lengths.get("left_forearm"), lengths.get("right_forearm")),
        ("Thigh     H-K", lengths.get("left_thigh"), lengths.get("right_thigh")),
        ("Shank     K-A", lengths.get("left_shank"), lengths.get("right_shank")),
    ]

    row_y = y + 72
    for label, left_value, right_value in rows:
        put_cv_text(image, label, x + 14, row_y, scale=0.38, color=(245, 245, 245))
        put_cv_text(image, fmt_short(left_value, digits=0), left_x, row_y, scale=0.39, color=(120, 255, 120))
        put_cv_text(image, fmt_short(right_value, digits=0), right_x, row_y, scale=0.39, color=(70, 220, 255))
        row_y += 21

    return image


def draw_video_status_bar(image, camera_fps=None, pose_fps=None, display_fps=None):
    """视频左下角轻量状态条。"""
    h, w = image.shape[:2]
    text = "Cam:{} | Pose:{} | GUI:{}".format(
        "--" if camera_fps is None else f"{camera_fps:.1f}",
        "--" if pose_fps is None else f"{pose_fps:.1f}",
        "--" if display_fps is None else f"{display_fps:.1f}",
    )
    x1, y1 = 8, h - 34
    x2, y2 = min(w - 8, 300), h - 8
    # 不再使用 overlay=image.copy() 做半透明条，降低每帧GUI绘制成本。
    cv2.rectangle(image, (x1, y1), (x2, y2), (16, 18, 22), -1)
    cv2.putText(image, text, (x1 + 10, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 245, 255), 1, cv2.LINE_AA)
    return image


# ============================================================
# 6. 运动学计算：长度、角度、速度
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
    points = {name: get_xyz(person_3d, name) for name in METRIC_KEYPOINT_NAMES}

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
# 7. CameraThread：只负责采集最新帧，不做YOLO和绘图
# ============================================================
class CameraThread(QThread):
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, frame_buffer: LatestBuffer, parent=None):
        super().__init__(parent)
        self.frame_buffer = frame_buffer
        self._running = False
        self.pipeline = None
        self.capture_fps = None
        self.frame_id = 0
        self.last_frame_time = None
        self.active_profile_desc = "未启动"

    def stop(self):
        self._running = False

    def _status(self, msg):
        print(f"[CameraThread] {msg}", flush=True)
        self.status_signal.emit(msg)

    def _error(self, msg):
        print(f"[CameraThread ERROR] {msg}", flush=True)
        self.error_signal.emit(msg)

    def _build_profile_candidates(self, depth_profile_list, color_profile_list):
        """
        构造候选profile组合。你这台 Femto Mega 日志显示 USB2.1，RGB888 很容易带宽过高，
        所以优先使用 MJPG 彩色流；如果拿不到帧，自动降级深度分辨率或改回原始配置。
        """
        y16 = fmt_name("Y16")
        mjpg = fmt_name("MJPG")
        rgb888 = fmt_name("RGB888")
        yuyv = fmt_name("YUYV")
        nv12 = fmt_name("NV12")

        depth_requests = [
            (REQUEST_DEPTH_W, REQUEST_DEPTH_H, y16, REQUEST_DEPTH_FPS, f"Depth {REQUEST_DEPTH_W}x{REQUEST_DEPTH_H}@{REQUEST_DEPTH_FPS} Y16"),
            (640, 576, y16, 30, "Depth 640x576@30 Y16"),
            (512, 512, y16, 30, "Depth 512x512@30 Y16"),
            (320, 288, y16, 30, "Depth 320x288@30 Y16"),
            (640, 576, y16, 15, "Depth 640x576@15 Y16"),
        ]
        color_requests_low_bw = [
            (REQUEST_COLOR_W, REQUEST_COLOR_H, mjpg, REQUEST_COLOR_FPS, f"Color {REQUEST_COLOR_W}x{REQUEST_COLOR_H}@{REQUEST_COLOR_FPS} MJPG"),
            (1280, 720, mjpg, 30, "Color 1280x720@30 MJPG"),
            (1280, 720, nv12, 30, "Color 1280x720@30 NV12"),
            (1280, 720, yuyv, 30, "Color 1280x720@30 YUYV"),
            (1280, 720, rgb888, 30, "Color 1280x720@30 RGB888"),
            # 保留一组你们原来能工作的配置作为兜底。
            (1920, 1080, rgb888, 30, "Color 1920x1080@30 RGB888(original)"),
        ]
        color_requests_normal = [
            (REQUEST_COLOR_W, REQUEST_COLOR_H, rgb888, REQUEST_COLOR_FPS, f"Color {REQUEST_COLOR_W}x{REQUEST_COLOR_H}@{REQUEST_COLOR_FPS} RGB888"),
            (REQUEST_COLOR_W, REQUEST_COLOR_H, mjpg, REQUEST_COLOR_FPS, f"Color {REQUEST_COLOR_W}x{REQUEST_COLOR_H}@{REQUEST_COLOR_FPS} MJPG"),
            (1280, 720, mjpg, 30, "Color 1280x720@30 MJPG"),
            (1920, 1080, rgb888, 30, "Color 1920x1080@30 RGB888(original)"),
        ]
        color_requests = color_requests_low_bw if LOW_BANDWIDTH_MODE else color_requests_normal

        depth_profiles = []
        for w, h, fmt, fps, desc in depth_requests:
            prof = get_profile_or_none(depth_profile_list, w, h, fmt, fps)
            if prof is not None:
                depth_profiles.append((prof, desc))

        color_profiles = []
        for w, h, fmt, fps, desc in color_requests:
            prof = get_profile_or_none(color_profile_list, w, h, fmt, fps)
            if prof is not None:
                color_profiles.append((prof, desc))

        # 最后保底：SDK默认profile。默认profile可能只有15FPS，但至少应该有画面。
        try:
            depth_profiles.append((depth_profile_list.get_default_video_stream_profile(), "Depth default"))
        except Exception:
            pass
        try:
            color_profiles.append((color_profile_list.get_default_video_stream_profile(), "Color default"))
        except Exception:
            pass

        # 组合顺序很重要：先试低带宽且30fps，再试兜底。
        pairs = []
        for d_prof, d_desc in depth_profiles:
            for c_prof, c_desc in color_profiles:
                key = (d_desc, c_desc)
                if key not in [(a, b) for _, a, _, b in pairs]:
                    pairs.append((d_prof, d_desc, c_prof, c_desc))
        return pairs

    def _try_start_once(self, depth_profile, depth_desc, color_profile, color_desc):
        pipeline = Pipeline()
        config = Config()
        config.enable_stream(depth_profile)
        config.enable_stream(color_profile)
        config.set_align_mode(OBAlignMode.HW_MODE)

        desc = f"RGB={color_desc} | Depth={depth_desc}"
        self._status(f"尝试启动相机profile：{desc}")
        pipeline.start(config)
        try:
            pipeline.enable_frame_sync()
            self._status("已开启 RGB 与 Depth 帧同步")
        except Exception as e:
            self._status(f"帧同步开启失败，继续运行：{e}")

        # 不要只看 pipeline.start 成功，必须确认真的能取到 RGB+Depth 帧。
        t0 = time.perf_counter()
        first_frames = None
        while self._running and time.perf_counter() - t0 < CAMERA_STARTUP_TIMEOUT_SEC:
            frames = pipeline.wait_for_frames(200)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is not None and depth_frame is not None:
                first_frames = frames
                break

        if first_frames is None:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise RuntimeError(f"该profile启动后 {CAMERA_STARTUP_TIMEOUT_SEC:.1f}s 内没有拿到RGB+Depth帧：{desc}")

        self.pipeline = pipeline
        self.active_profile_desc = desc
        self._status(f"相机启动成功：{desc}")
        return first_frames

    def _frames_to_bundle(self, frames, fx, fy, cx, cy):
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        color_image = decode_color_frame(color_frame)

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_h = depth_frame.get_height()
        depth_w = depth_frame.get_width()
        depth_matrix = depth_data.reshape((depth_h, depth_w)).copy()

        try:
            depth_scale = depth_frame.get_depth_scale()
        except Exception:
            depth_scale = 1.0

        self.frame_id += 1
        now_ts = time.time()
        self.last_frame_time = now_ts
        return FrameBundle(
            frame_id=self.frame_id,
            timestamp=now_ts,
            color_image=color_image,
            depth_matrix=depth_matrix,
            depth_scale=depth_scale,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    def run(self):
        self._running = True
        self.pipeline = None
        try:
            # 先用临时pipeline查询profile列表。
            probe = Pipeline()
            depth_profile_list = probe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            color_profile_list = probe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            profile_pairs = self._build_profile_candidates(depth_profile_list, color_profile_list)
            try:
                probe.stop()
            except Exception:
                pass

            if not profile_pairs:
                self._error("没有可用的RGB/Depth profile")
                return

            first_frames = None
            last_error = None
            for depth_profile, depth_desc, color_profile, color_desc in profile_pairs:
                if not self._running:
                    return
                try:
                    first_frames = self._try_start_once(depth_profile, depth_desc, color_profile, color_desc)
                    break
                except Exception as e:
                    last_error = e
                    self._status(f"当前profile不可用，切换下一组：{e}")
                    self.pipeline = None
                    continue

            if self.pipeline is None or first_frames is None:
                self._error(f"所有相机profile都无法输出画面，最后错误：{last_error}")
                return

            camera_param = self.pipeline.get_camera_param()
            rgb_intrinsic = camera_param.rgb_intrinsic
            fx, fy, cx, cy = rgb_intrinsic.fx, rgb_intrinsic.fy, rgb_intrinsic.cx, rgb_intrinsic.cy
            self._status(f"RGB内参：fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

            # 先把启动阶段拿到的第一帧送出去，避免界面一直黑。
            try:
                bundle = self._frames_to_bundle(first_frames, fx, fy, cx, cy)
                if bundle is not None:
                    self.frame_buffer.set(bundle)
            except Exception as e:
                self._status(f"第一帧解码失败，继续采集：{e}")

            fps_t0 = time.perf_counter()
            fps_count = 0

            while self._running:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                try:
                    bundle = self._frames_to_bundle(frames, fx, fy, cx, cy)
                    if bundle is None:
                        continue
                    self.frame_buffer.set(bundle)
                    fps_count += 1
                except Exception as e:
                    # 这里一定要暴露出来，不然界面会显示“测量中”但没有画面。
                    self._status(f"帧解码/打包失败：{e}")
                    continue

                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    self.capture_fps = fps_count / (now - fps_t0)
                    fps_t0 = now
                    fps_count = 0

        except Exception as e:
            self._error(f"相机线程错误：{e}")

        finally:
            try:
                if self.pipeline is not None:
                    self.pipeline.stop()
            except Exception:
                pass
            self._status("相机已关闭")

# ============================================================
# 8. PoseThread：只负责YOLO + 3D计算，不负责GUI刷新
# ============================================================
class PoseProcessThread(QThread):
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, frame_buffer: LatestBuffer, pose_buffer: LatestBuffer, parent=None):
        super().__init__(parent)
        self.frame_buffer = frame_buffer
        self.pose_buffer = pose_buffer
        self._running = False
        self.pose_model = None
        self.prev_state = {"time": None, "points": {}, "angles": {}}
        self.pose_fps = None
        self.last_processed_frame_id = -1
        self.device = "cpu"
        self.use_half = False

    def stop(self):
        self._running = False

    def _reset_velocity_state(self):
        self.prev_state["time"] = None
        self.prev_state["points"] = {}
        self.prev_state["angles"] = {}

    def _load_model(self):
        self.status_signal.emit("正在加载 YOLOv8 Pose 模型...")

        # OpenCV内部线程太多有时会和PyQt/torch抢CPU，限制一下更稳定。
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                self.device = 0
                self.use_half = bool(ENABLE_CUDA_HALF)
                try:
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
                self.status_signal.emit("检测到 CUDA：YOLO使用GPU" + (" + FP16" if self.use_half else ""))
            else:
                self.device = "cpu"
                self.use_half = False
                self.status_signal.emit("未检测到 CUDA：YOLO使用CPU")
        except Exception:
            self.device = "cpu"
            self.use_half = False
            self.status_signal.emit("无法检查CUDA：YOLO使用CPU")

        self.pose_model = YOLO(MODEL_PATH)
        try:
            self.pose_model.fuse()
        except Exception:
            pass

        # 预热一次，避免第一帧特别慢。
        try:
            dummy = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)
            self.pose_model.predict(
                dummy,
                verbose=False,
                imgsz=YOLO_IMGSZ,
                device=self.device,
                half=self.use_half,
                conf=YOLO_CONF_THRES,
                max_det=MAX_DET,
            )
        except Exception:
            pass

        self.status_signal.emit("YOLO模型加载完成")

    def run(self):
        self._running = True
        try:
            self._load_model()
        except Exception as e:
            self.error_signal.emit(f"YOLO模型加载失败：{e}")
            return

        fps_t0 = time.perf_counter()
        fps_count = 0

        while self._running:
            bundle = self.frame_buffer.get()
            if bundle is None:
                time.sleep(0.002)
                continue

            # 只处理最新帧；同一帧不重复处理。
            if bundle.frame_id == self.last_processed_frame_id:
                time.sleep(0.001)
                continue

            self.last_processed_frame_id = bundle.frame_id

            try:
                pose_result = self.process_one_frame(bundle)
                self.pose_buffer.set(pose_result)

                fps_count += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    self.pose_fps = fps_count / (now - fps_t0)
                    fps_t0 = now
                    fps_count = 0

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

        # 推理线程只resize一次；GUI显示也resize，但二者解耦。
        proc_color = cv2.resize(color_image, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)

        results = self.pose_model.predict(
            proc_color,
            verbose=False,
            imgsz=YOLO_IMGSZ,
            device=self.device,
            half=self.use_half,
            conf=YOLO_CONF_THRES,
            max_det=MAX_DET,
        )
        result = results[0]

        person_3d = None
        metrics = None
        selected_score = None
        selected_box = None
        kp_xy_proc = None
        kp_conf = None
        has_person = False

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
                has_person = True

                if boxes_xyxy is not None and selected_index < len(boxes_xyxy):
                    selected_box = boxes_xyxy[selected_index]

                person_3d = {"id": 0, "keypoints": {name: None for name in COCO_KEYPOINT_NAMES}}

                # 只对运动学需要的12个关键点取深度，节省时间。
                for name in METRIC_KEYPOINT_NAMES:
                    kpt_id = COCO_KEYPOINT_NAMES.index(name)
                    x_proc, y_proc = kp_xy_proc[kpt_id]
                    conf = float(kp_conf[kpt_id])

                    if conf < KEYPOINT_CONF_THRES or x_proc <= 0 or y_proc <= 0:
                        person_3d["keypoints"][name] = None
                        continue

                    # YOLO在proc_color上识别，映射回原始RGB，再映射到depth。
                    u_rgb = x_proc * color_w / PROC_W
                    v_rgb = y_proc * color_h / PROC_H
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

                metrics = compute_metrics(person_3d, self.prev_state, bundle.timestamp)

        if not has_person:
            self._reset_velocity_state()

        return PoseResultBundle(
            frame_id=bundle.frame_id,
            timestamp=time.time(),
            kp_xy_proc=kp_xy_proc,
            kp_conf=kp_conf,
            box_xyxy=selected_box,
            score=selected_score,
            person_3d=person_3d,
            metrics=metrics,
            has_person=has_person,
        )


# ============================================================
# 9. PyQt 自绘曲线卡片
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
        self.series = series
        self.fixed_ylim = fixed_ylim
        self.force_zero_min = force_zero_min
        self.max_len = HISTORY_LEN
        self.history = {key: deque(maxlen=self.max_len) for key, _, _ in series}
        self.latest = {key: np.nan for key, _, _ in series}

        self.setMinimumSize(QSize(245, 135))
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
        painter.setRenderHint(QPainter.Antialiasing, False)

        rect = self.rect().adjusted(4, 4, -4, -4)
        bg = QColor("#eaf7ef")
        border = QColor("#93b9a0")
        title_color = QColor("#143b2a")
        grid_color = QColor("#c9dfd1")
        axis_color = QColor("#7b9282")
        text_dim = QColor("#5c6d62")

        painter.setPen(QPen(border, 1))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(rect, 8, 8)

        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        painter.setPen(title_color)
        painter.drawText(rect.left() + 10, rect.top() + 21, self.title)

        painter.setFont(QFont("Arial", 8))
        painter.setPen(text_dim)
        painter.drawText(rect.right() - 54, rect.top() + 20, self.unit)

        plot_left = rect.left() + 42
        plot_top = rect.top() + 34
        plot_right = rect.right() - 8
        plot_bottom = rect.bottom() - 16
        plot_w = max(1, plot_right - plot_left)
        plot_h = max(1, plot_bottom - plot_top)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#f8fffa")))
        painter.drawRect(plot_left, plot_top, plot_w, plot_h)

        y_min, y_max = self._get_ylim()
        y_span = max(1e-6, y_max - y_min)

        painter.setFont(QFont("Arial", 7))
        for i in range(4):
            yy = plot_top + int(i * plot_h / 3)
            val = y_max - i * y_span / 3
            painter.setPen(QPen(grid_color, 1))
            painter.drawLine(plot_left, yy, plot_right, yy)
            painter.setPen(text_dim)
            painter.drawText(rect.left() + 7, yy + 4, f"{val:.0f}")

        painter.setPen(QPen(axis_color, 1))
        painter.drawLine(plot_left, plot_top, plot_left, plot_bottom)
        painter.drawLine(plot_left, plot_bottom, plot_right, plot_bottom)

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
            painter.drawText(plot_left + 10, plot_top + 26, "等待数据...")

        # 图例与当前数值
        legend_x = plot_right - 120
        legend_y = plot_top + 5
        legend_w = 116
        legend_h = 15 * len(self.series) + 7
        painter.setPen(QPen(QColor("#d6eadc"), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255, 225)))
        painter.drawRoundedRect(legend_x, legend_y, legend_w, legend_h, 5, 5)

        painter.setFont(QFont("Arial", 7))
        for idx, (key, label, color) in enumerate(self.series):
            yy = legend_y + 14 + idx * 15
            painter.setPen(QPen(color, 2))
            painter.drawLine(legend_x + 5, yy - 4, legend_x + 18, yy - 4)
            painter.setPen(color)
            painter.drawText(legend_x + 22, yy, f"{label}:{fmt_value(self.latest.get(key), 0)}")

        painter.end()



# ============================================================
# 10. 快速视频显示控件：避免每帧 QPixmap.scaled()
# ============================================================
class VideoDisplayLabel(QLabel):
    """
    QLabel 的轻量替代：
    1. 主线程只把 BGR 转 RGB，保存为 numpy 缓存；
    2. paintEvent 中用 QPainter 按保持比例绘制 QImage；
    3. 避免每帧 QPixmap.fromImage + pixmap.scaled 的额外开销。
    """

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._rgb_cache = None
        self._qimage = None
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    def clear_image(self):
        self._rgb_cache = None
        self._qimage = None
        self.update()

    def set_bgr_image(self, image_bgr):
        if image_bgr is None:
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._rgb_cache = np.ascontiguousarray(rgb)
        h, w, ch = self._rgb_cache.shape
        bytes_per_line = ch * w
        self._qimage = QImage(self._rgb_cache.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setText("")
        self.update()

    def paintEvent(self, event):
        if self._qimage is None:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.fillRect(self.rect(), QColor("#151923"))

        img_w = self._qimage.width()
        img_h = self._qimage.height()
        if img_w <= 0 or img_h <= 0:
            painter.end()
            return

        scale = min(self.width() / img_w, self.height() / img_h)
        target_w = int(img_w * scale)
        target_h = int(img_h * scale)
        x = (self.width() - target_w) // 2
        y = (self.height() - target_h) // 2
        target_rect = QRect(x, y, target_w, target_h)
        painter.drawImage(target_rect, self._qimage)
        painter.end()

# ============================================================
# 10. 主界面
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("康复视觉评定系统 - 优化多线程版")
        self.resize(1420, 860)

        self.frame_buffer = LatestBuffer()
        self.pose_buffer = LatestBuffer()
        self.camera_thread = None
        self.pose_thread = None
        self.running = False

        self.display_fps = None
        self._display_fps_count = 0
        self._display_fps_t0 = time.perf_counter()
        self._last_plotted_pose_frame_id = -1
        self._last_displayed_frame_id = -1
        self.last_status_message = ""
        self.last_error_message = ""

        self._build_ui()
        self._apply_style()
        self._build_timers()

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

        left_panel = QFrame()
        left_panel.setObjectName("MainPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(12)

        self.video_label = VideoDisplayLabel("点击“开始测量”后显示实时视频")
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

        right_panel = QFrame()
        right_panel.setObjectName("MainPanel")
        right_layout = QGridLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setHorizontalSpacing(10)
        right_layout.setVerticalSpacing(10)

        self.cards = {}
        self._create_plot_cards()

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
            "右关节角度曲线", "deg",
            [("right_shoulder", "肩", blue), ("right_elbow", "肘", green), ("right_knee", "膝", orange)],
            fixed_ylim=(0.0, 180.0),
        )
        self.cards["left_angles"] = PlotCard(
            "左关节角度曲线", "deg",
            [("left_shoulder", "肩", blue), ("left_elbow", "肘", green), ("left_knee", "膝", orange)],
            fixed_ylim=(0.0, 180.0),
        )
        self.cards["right_linear_velocity"] = PlotCard(
            "右线速度曲线", "mm/s",
            [("right_wrist", "腕", purple), ("right_ankle", "踝", cyan)],
            fixed_ylim=None,
        )
        self.cards["left_linear_velocity"] = PlotCard(
            "左线速度曲线", "mm/s",
            [("left_wrist", "腕", purple), ("left_ankle", "踝", cyan)],
            fixed_ylim=None,
        )
        self.cards["right_angular_velocity"] = PlotCard(
            "右角速度曲线", "deg/s",
            [("right_shoulder", "肩", blue), ("right_elbow", "肘", green), ("right_knee", "膝", orange)],
            fixed_ylim=None,
        )
        self.cards["left_angular_velocity"] = PlotCard(
            "左角速度曲线", "deg/s",
            [("left_shoulder", "肩", blue), ("left_elbow", "肘", green), ("left_knee", "膝", orange)],
            fixed_ylim=None,
        )
        self.cards["right_lengths"] = PlotCard(
            "右肢段长度测量", "mm",
            [("right_upper_arm", "上臂", blue), ("right_forearm", "前臂", green), ("right_thigh", "大腿", orange), ("right_shank", "小腿", red)],
            fixed_ylim=None,
        )
        self.cards["left_lengths"] = PlotCard(
            "左肢段长度测量", "mm",
            [("left_upper_arm", "上臂", blue), ("left_forearm", "前臂", green), ("left_thigh", "大腿", orange), ("left_shank", "小腿", red)],
            fixed_ylim=None,
        )

    def _build_timers(self):
        self.video_timer = QTimer(self)
        self.video_timer.setTimerType(Qt.PreciseTimer)
        self.video_timer.timeout.connect(self.update_video_from_latest)

        self.plot_timer = QTimer(self)
        self.plot_timer.setTimerType(Qt.PreciseTimer)
        self.plot_timer.timeout.connect(self.update_cards_from_latest)

        self.status_timer = QTimer(self)
        self.status_timer.setTimerType(Qt.PreciseTimer)
        self.status_timer.timeout.connect(self.update_status_text)

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

        self.frame_buffer.clear()
        self.pose_buffer.clear()
        self.clear_all_cards()
        self._last_plotted_pose_frame_id = -1
        self._last_displayed_frame_id = -1
        self.last_status_message = ""
        self.last_error_message = ""
        self.video_label.clear_image()
        self.video_label.setText("正在启动相机与模型，请稍等...")
        self.status_label.setText("状态：正在启动")

        self.camera_thread = CameraThread(self.frame_buffer)
        self.pose_thread = PoseProcessThread(self.frame_buffer, self.pose_buffer)

        self.camera_thread.status_signal.connect(self.on_status)
        self.camera_thread.error_signal.connect(self.on_error)
        self.pose_thread.status_signal.connect(self.on_status)
        self.pose_thread.error_signal.connect(self.on_error)

        # 先启动相机和模型均可。YOLO线程会等到frame_buffer中出现最新帧。
        self.camera_thread.start()
        self.pose_thread.start()

        self.video_timer.start(VIDEO_TIMER_MS)
        self.plot_timer.start(PLOT_TIMER_MS)
        self.status_timer.start(STATUS_TIMER_MS)

        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_measurement(self):
        if not self.running:
            return

        self.video_timer.stop()
        self.plot_timer.stop()
        self.status_timer.stop()

        self.status_label.setText("状态：正在结束测量...")
        QApplication.processEvents()

        if self.camera_thread is not None:
            self.camera_thread.stop()
        if self.pose_thread is not None:
            self.pose_thread.stop()

        if self.camera_thread is not None:
            self.camera_thread.wait(2000)
        if self.pose_thread is not None:
            self.pose_thread.wait(2000)

        self.camera_thread = None
        self.pose_thread = None
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("状态：测量已结束，曲线保留为最后一次结果")

    def closeEvent(self, event):
        self.stop_measurement()
        event.accept()

    def on_status(self, msg):
        self.last_status_message = str(msg)
        print(f"[UI STATUS] {msg}", flush=True)
        self.status_label.setText(f"状态：{msg}")

    def on_error(self, msg):
        self.last_error_message = str(msg)
        print(f"[UI ERROR] {msg}", flush=True)
        self.status_label.setText(f"错误：{msg}")

    def update_video_from_latest(self):
        bundle = self.frame_buffer.get()
        if bundle is None:
            return
        # 没有新相机帧就不重复做 cv2.resize / QImage / paint，避免GUI空转。
        if bundle.frame_id == self._last_displayed_frame_id:
            return
        self._last_displayed_frame_id = bundle.frame_id

        # 只复制显示图，不改原始frame_buffer里的图。
        display_bgr = cv2.resize(bundle.color_image, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)

        pose = self.pose_buffer.get()
        pose_valid = False
        if pose is not None and pose.has_person:
            if time.time() - pose.timestamp <= POSE_RESULT_TTL:
                pose_valid = True

        if pose_valid:
            if SHOW_SKELETON_ON_VIDEO:
                display_bgr = draw_selected_pose(
                    display_bgr,
                    pose.kp_xy_proc,
                    pose.kp_conf,
                    box_xyxy=pose.box_xyxy,
                    score=pose.score,
                    conf_thres=KEYPOINT_CONF_THRES,
                )

            if SHOW_3D_COORDS:
                display_bgr = draw_3d_coords_near_keypoints(
                    display_bgr,
                    pose.person_3d,
                    show_all_keypoints=SHOW_ALL_KEYPOINTS,
                    important_keypoints=IMPORTANT_KEYPOINTS,
                )

            if SHOW_BONE_LENGTH_ON_VIDEO:
                display_bgr = draw_bone_length_on_video(display_bgr, pose.metrics, score=pose.score)
        else:
            if SHOW_BONE_LENGTH_ON_VIDEO:
                display_bgr = draw_bone_length_on_video(display_bgr, None, score=None)

        if SHOW_VIDEO_STATUS_BAR:
            camera_fps = self.camera_thread.capture_fps if self.camera_thread is not None else None
            pose_fps = self.pose_thread.pose_fps if self.pose_thread is not None else None
            display_bgr = draw_video_status_bar(display_bgr, camera_fps=camera_fps, pose_fps=pose_fps, display_fps=self.display_fps)

        self.update_video_label(display_bgr)

        self._display_fps_count += 1
        now = time.perf_counter()
        if now - self._display_fps_t0 >= 1.0:
            self.display_fps = self._display_fps_count / (now - self._display_fps_t0)
            self._display_fps_t0 = now
            self._display_fps_count = 0

    def update_video_label(self, image_bgr):
        if image_bgr is None:
            return
        # 使用自定义 VideoDisplayLabel，避免每帧 QPixmap.scaled()。
        self.video_label.set_bgr_image(image_bgr)

    def update_cards_from_latest(self):
        pose = self.pose_buffer.get()
        if pose is None:
            return

        # 没有新姿态结果就不重复压入曲线，避免同一个数据点被重复画多次。
        if pose.frame_id == self._last_plotted_pose_frame_id:
            return
        self._last_plotted_pose_frame_id = pose.frame_id

        metrics = pose.metrics if pose.has_person else None
        if metrics is None:
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

    def update_status_text(self):
        if self.last_error_message:
            self.status_label.setText(f"错误：{self.last_error_message}")
            return

        cam = self.camera_thread.capture_fps if self.camera_thread is not None else None
        pose = self.pose_thread.pose_fps if self.pose_thread is not None else None
        gui = self.display_fps
        latest_pose = self.pose_buffer.get()
        score = latest_pose.score if latest_pose is not None else None
        latest_frame = self.frame_buffer.get()
        profile = self.camera_thread.active_profile_desc if self.camera_thread is not None else "--"

        if latest_frame is None:
            base = self.last_status_message or "正在等待相机第一帧"
            self.status_label.setText(f"状态：{base} | Camera=-- FPS | Pose=-- FPS | GUI=-- FPS")
            return

        self.status_label.setText(
            "状态：测量中 | Camera={} FPS | Pose={} FPS | GUI={} FPS | ID0 conf={} | {}".format(
                "--" if cam is None else f"{cam:.1f}",
                "--" if pose is None else f"{pose:.1f}",
                "--" if gui is None else f"{gui:.1f}",
                "--" if score is None else f"{score:.2f}",
                profile,
            )
        )


# ============================================================
# 11. 入口
# ============================================================
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

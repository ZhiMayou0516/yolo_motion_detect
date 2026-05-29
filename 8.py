import cv2
import time
from collections import deque

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

# YOLO Pose 骨架连接关系，使用 COCO 关键点编号
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
# 4. 绘图辅助函数
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


def draw_selected_pose(image, kp_xy_proc, kp_conf, box_xyxy=None, score=None, conf_thres=0.30):
    """
    只画当前选中的人：固定显示为 ID:0。
    选中逻辑在主循环里完成：原始检测结果中置信度最高的人作为 ID0。
    """
    if kp_xy_proc is None or kp_conf is None:
        return image

    # 画骨架线
    for a, b in SKELETON_PAIRS:
        if a >= len(kp_xy_proc) or b >= len(kp_xy_proc):
            continue
        if kp_conf[a] < conf_thres or kp_conf[b] < conf_thres:
            continue

        xa, ya = kp_xy_proc[a]
        xb, yb = kp_xy_proc[b]
        if xa <= 0 or ya <= 0 or xb <= 0 or yb <= 0:
            continue

        cv2.line(
            image,
            (int(round(xa)), int(round(ya))),
            (int(round(xb)), int(round(yb))),
            (0, 255, 0),
            2
        )

    # 画关键点
    for kpt_id, (x, y) in enumerate(kp_xy_proc):
        if kp_conf[kpt_id] < conf_thres:
            continue
        if x <= 0 or y <= 0:
            continue
        cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)

    # 画检测框和 ID0
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = "ID:0"
        if score is not None:
            label += f"  conf:{score:.2f}"
        draw_text_with_background(
            image,
            label,
            x1,
            max(22, y1 - 8),
            font_scale=0.60,
            text_color=(0, 0, 255),
            thickness=2
        )

    return image


def draw_3d_coords_near_keypoints(image, person_3d, show_all_keypoints=True, important_keypoints=None):
    """
    将 ID0 每个关键点的3D坐标直接画在对应点旁边。
    """
    if person_3d is None or len(person_3d.get("keypoints", {})) == 0:
        draw_text_with_background(
            image,
            "No ID0 person detected",
            20,
            35,
            font_scale=0.7,
            text_color=(0, 255, 255),
            thickness=2
        )
        return image

    if important_keypoints is None:
        important_keypoints = set()

    keypoints = person_3d["keypoints"]

    # 为了避免所有文字往一个方向堆，用不同点位设置不同偏移
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

    default_offset = (8, -8)

    for name, value in keypoints.items():
        if not show_all_keypoints and name not in important_keypoints:
            continue
        if value is None:
            continue

        x_proc, y_proc = value["pixel_proc"]
        X, Y, Z = value["xyz_mm"]

        short_name = DISPLAY_NAMES.get(name, name)
        text = f"{short_name}:({X:.0f},{Y:.0f},{Z:.0f})"

        px = int(round(x_proc))
        py = int(round(y_proc))

        cv2.circle(image, (px, py), 4, (0, 255, 255), -1)

        offset_x, offset_y = offset_map.get(name, default_offset)
        draw_text_with_background(
            image,
            text,
            px + offset_x,
            py + offset_y,
            font_scale=0.34,
            text_color=(0, 255, 255),
            thickness=1
        )

    return image


def fmt_value(value, unit="", digits=1):
    if value is None:
        return "--"
    try:
        return f"{value:.{digits}f}{unit}"
    except Exception:
        return "--"


def draw_metrics_panel(image, metrics, score=None):
    """
    在 OpenCV 图像左上角实时显示所有计算参数。
    为了避免 OpenCV 中文乱码，这里用英文缩写显示。
    """
    h, w = image.shape[:2]

    panel_x1, panel_y1 = 8, 8
    panel_w = min(625, w - 16)
    panel_h = 342
    panel_x2 = panel_x1 + panel_w
    panel_y2 = min(panel_y1 + panel_h, h - 8)

    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)
    cv2.rectangle(image, (panel_x1, panel_y1), (panel_x2, panel_y2), (255, 255, 255), 1)

    x = panel_x1 + 10
    y = panel_y1 + 22
    line_h = 18
    font = cv2.FONT_HERSHEY_SIMPLEX

    title = "ID0 = highest confidence person"
    if score is not None:
        title += f" | conf={score:.2f}"
    cv2.putText(image, title, (x, y), font, 0.48, (0, 255, 255), 1)
    y += line_h + 2

    if metrics is None:
        cv2.putText(image, "No valid metrics", (x, y), font, 0.48, (0, 255, 255), 1)
        return image

    def put(line, color=(255, 255, 255), scale=0.42):
        nonlocal y
        if y > panel_y2 - 8:
            return
        cv2.putText(image, line, (x, y), font, scale, color, 1)
        y += line_h

    lengths = metrics.get("lengths", {})
    angles = metrics.get("angles", {})
    lin_vel = metrics.get("linear_velocity", {})
    ang_vel = metrics.get("angular_velocity", {})

    put("[Bone length] unit:mm", (0, 255, 255))
    put(
        "UpperArm S-E   L:{}  R:{}".format(
            fmt_value(lengths.get("left_upper_arm")),
            fmt_value(lengths.get("right_upper_arm"))
        )
    )
    put(
        "ForeArm  E-W   L:{}  R:{}".format(
            fmt_value(lengths.get("left_forearm")),
            fmt_value(lengths.get("right_forearm"))
        )
    )
    put(
        "Thigh    H-K   L:{}  R:{}".format(
            fmt_value(lengths.get("left_thigh")),
            fmt_value(lengths.get("right_thigh"))
        )
    )
    put(
        "Shank    K-A   L:{}  R:{}".format(
            fmt_value(lengths.get("left_shank")),
            fmt_value(lengths.get("right_shank"))
        )
    )

    y += 3
    put("[Joint angle] unit:deg", (0, 255, 255))
    put(
        "Shoulder E-S-H L:{}  R:{}".format(
            fmt_value(angles.get("left_shoulder")),
            fmt_value(angles.get("right_shoulder"))
        )
    )
    put(
        "Elbow    S-E-W L:{}  R:{}".format(
            fmt_value(angles.get("left_elbow")),
            fmt_value(angles.get("right_elbow"))
        )
    )
    put(
        "Knee     H-K-A L:{}  R:{}".format(
            fmt_value(angles.get("left_knee")),
            fmt_value(angles.get("right_knee"))
        )
    )

    y += 3
    put("[Linear velocity] unit:mm/s", (0, 255, 255))
    put(
        "Wrist          L:{}  R:{}".format(
            fmt_value(lin_vel.get("left_wrist")),
            fmt_value(lin_vel.get("right_wrist"))
        )
    )
    put(
        "Ankle          L:{}  R:{}".format(
            fmt_value(lin_vel.get("left_ankle")),
            fmt_value(lin_vel.get("right_ankle"))
        )
    )

    y += 3
    put("[Angular velocity] unit:deg/s", (0, 255, 255))
    put(
        "Shoulder       L:{}  R:{}".format(
            fmt_value(ang_vel.get("left_shoulder")),
            fmt_value(ang_vel.get("right_shoulder"))
        )
    )
    put(
        "Elbow          L:{}  R:{}".format(
            fmt_value(ang_vel.get("left_elbow")),
            fmt_value(ang_vel.get("right_elbow"))
        )
    )
    put(
        "Knee           L:{}  R:{}".format(
            fmt_value(ang_vel.get("left_knee")),
            fmt_value(ang_vel.get("right_knee"))
        )
    )

    return image



# ============================================================
# 5. 独立运动学仪表盘窗口：紧凑深色风格，数值 + 波形
# ============================================================
# 颜色为 OpenCV BGR
ANGLE_SERIES = [
    ("left_shoulder", "L-Shoulder", (0, 165, 255)),
    ("right_shoulder", "R-Shoulder", (0, 190, 255)),
    ("left_elbow", "L-Elbow", (70, 255, 70)),
    ("right_elbow", "R-Elbow", (0, 255, 0)),
    ("left_knee", "L-Knee", (255, 255, 0)),
    ("right_knee", "R-Knee", (255, 230, 0)),
]

LINEAR_VEL_SERIES = [
    ("left_wrist", "L-Wrist", (0, 165, 255)),
    ("right_wrist", "R-Wrist", (0, 190, 255)),
    ("left_ankle", "L-Ankle", (255, 255, 0)),
    ("right_ankle", "R-Ankle", (255, 230, 0)),
]

ANGULAR_VEL_SERIES = [
    ("left_shoulder", "L-Shoulder", (0, 165, 255)),
    ("right_shoulder", "R-Shoulder", (0, 190, 255)),
    ("left_elbow", "L-Elbow", (70, 255, 70)),
    ("right_elbow", "R-Elbow", (0, 255, 0)),
    ("left_knee", "L-Knee", (255, 255, 0)),
    ("right_knee", "R-Knee", (255, 230, 0)),
]


DASHBOARD_WINDOW_NAME = "Kinematics Dashboard"
DASHBOARD_W = 640
DASHBOARD_H = 720

DASH_BG = (24, 21, 22)
PANEL_BG = (27, 23, 24)
PANEL_BORDER = (95, 70, 72)
GRID_COLOR = (52, 43, 45)
TEXT_MAIN = (245, 245, 245)
TEXT_DIM = (150, 145, 145)
TEXT_WARN = (70, 210, 255)


def init_motion_history(max_len=120):
    """
    保存最近 max_len 帧的运动学参数，用于独立窗口中的实时波形图。
    120帧左右足够显示趋势，窗口也不会被横向拉得太长。
    """
    return {
        "angles": {key: deque(maxlen=max_len) for key, _, _ in ANGLE_SERIES},
        "linear_velocity": {key: deque(maxlen=max_len) for key, _, _ in LINEAR_VEL_SERIES},
        "angular_velocity": {key: deque(maxlen=max_len) for key, _, _ in ANGULAR_VEL_SERIES},
    }


def safe_float_or_nan(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
        if not np.isfinite(value):
            return np.nan
        return value
    except Exception:
        return np.nan


def update_motion_history(history, metrics):
    """
    每一帧更新一次历史数据。
    没检测到人或某个点无效时写入 NaN，波形会自然断开。
    """
    for group_name, series in [
        ("angles", ANGLE_SERIES),
        ("linear_velocity", LINEAR_VEL_SERIES),
        ("angular_velocity", ANGULAR_VEL_SERIES),
    ]:
        group_values = {}
        if metrics is not None:
            group_values = metrics.get(group_name, {}) or {}

        for key, _, _ in series:
            history[group_name][key].append(safe_float_or_nan(group_values.get(key)))


def put_cv_text(
    image,
    text,
    x,
    y,
    scale=0.46,
    color=TEXT_MAIN,
    thickness=1,
):
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


def finite_latest(dq):
    if dq is None or len(dq) == 0:
        return None
    v = dq[-1]
    if not np.isfinite(v):
        return None
    return float(v)


def get_finite_values_from_history(history_group):
    values = []
    for dq in history_group.values():
        for v in dq:
            if np.isfinite(v):
                values.append(float(v))
    return values


def draw_dark_panel(image, x, y, w, h, title):
    """画一个紧凑深色卡片。"""
    cv2.rectangle(image, (x, y), (x + w, y + h), PANEL_BG, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), PANEL_BORDER, 1)
    put_cv_text(image, title, x + 12, y + 27, scale=0.58, color=TEXT_MAIN, thickness=2)


def draw_bone_length_compact(image, x, y, w, h, lengths):
    """紧凑显示骨长，不再占很宽的表格。"""
    draw_dark_panel(image, x, y, w, h, "0. Bone Length (3D, mm)")

    rows = [
        ("Upper arm  S-E", lengths.get("left_upper_arm"), lengths.get("right_upper_arm")),
        ("Forearm    E-W", lengths.get("left_forearm"), lengths.get("right_forearm")),
        ("Thigh      H-K", lengths.get("left_thigh"), lengths.get("right_thigh")),
        ("Shank      K-A", lengths.get("left_shank"), lengths.get("right_shank")),
    ]

    put_cv_text(image, "Item", x + 18, y + 54, scale=0.40, color=TEXT_DIM)
    put_cv_text(image, "Left", x + int(w * 0.56), y + 54, scale=0.40, color=TEXT_DIM)
    put_cv_text(image, "Right", x + int(w * 0.75), y + 54, scale=0.40, color=TEXT_DIM)

    row_y = y + 78
    for label, left_value, right_value in rows:
        put_cv_text(image, label, x + 18, row_y, scale=0.43, color=TEXT_MAIN)
        put_cv_text(image, fmt_short(left_value, digits=0), x + int(w * 0.56), row_y, scale=0.43, color=(120, 255, 120))
        put_cv_text(image, fmt_short(right_value, digits=0), x + int(w * 0.75), row_y, scale=0.43, color=(70, 220, 255))
        row_y += 23




def draw_bone_length_on_color(image, metrics, score=None):
    """
    将骨长参数单独叠加到 RGB/color 窗口上。
    独立运动学窗口只保留角度、线速度、角速度波形，避免骨长表格显示不全。
    """
    h, w = image.shape[:2]

    lengths = {}
    if metrics is not None:
        lengths = metrics.get("lengths", {}) or {}

    panel_w = min(390, max(320, w - 20))
    panel_h = 158
    x = 10
    y = 10

    # 半透明深色底板
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (18, 16, 17), -1)
    cv2.addWeighted(overlay, 0.66, image, 0.34, 0, image)
    cv2.rectangle(image, (x, y), (x + panel_w, y + panel_h), (115, 90, 92), 1)

    title = "Bone Length (3D, mm)"
    if score is not None:
        title += f"   conf:{score:.2f}"
    put_cv_text(image, title, x + 12, y + 25, scale=0.52, color=TEXT_MAIN, thickness=2)

    left_x = x + int(panel_w * 0.58)
    right_x = x + int(panel_w * 0.78)
    put_cv_text(image, "Item", x + 14, y + 52, scale=0.38, color=TEXT_DIM)
    put_cv_text(image, "Left", left_x, y + 52, scale=0.38, color=TEXT_DIM)
    put_cv_text(image, "Right", right_x, y + 52, scale=0.38, color=TEXT_DIM)

    rows = [
        ("Upper arm  S-E", lengths.get("left_upper_arm"), lengths.get("right_upper_arm")),
        ("Forearm    E-W", lengths.get("left_forearm"), lengths.get("right_forearm")),
        ("Thigh      H-K", lengths.get("left_thigh"), lengths.get("right_thigh")),
        ("Shank      K-A", lengths.get("left_shank"), lengths.get("right_shank")),
    ]

    row_y = y + 78
    for label, left_value, right_value in rows:
        put_cv_text(image, label, x + 14, row_y, scale=0.42, color=TEXT_MAIN)
        put_cv_text(image, fmt_short(left_value, digits=0), left_x, row_y, scale=0.43, color=(120, 255, 120), thickness=1)
        put_cv_text(image, fmt_short(right_value, digits=0), right_x, row_y, scale=0.43, color=(70, 220, 255), thickness=1)
        row_y += 23

    return image


def draw_dark_series_plot(
    image,
    x,
    y,
    w,
    h,
    title,
    history_group,
    series_info,
    unit_text,
    fixed_ylim=None,
    force_zero_min=True,
):
    """
    紧凑版实时波形图：暗色背景、短横向窗口、右上角直接显示当前数值。
    """
    draw_dark_panel(image, x, y, w, h, title)

    plot_x1 = x + 58
    plot_y1 = y + 46
    plot_x2 = x + w - 14
    plot_y2 = y + h - 18
    plot_w = plot_x2 - plot_x1
    plot_h = plot_y2 - plot_y1

    # 绘图区
    cv2.rectangle(image, (plot_x1, plot_y1), (plot_x2, plot_y2), (26, 22, 23), -1)

    if fixed_ylim is not None:
        y_min, y_max = fixed_ylim
    else:
        values = get_finite_values_from_history(history_group)
        if len(values) == 0:
            y_min, y_max = (0.0, 1.0)
        else:
            y_min = min(values)
            y_max = max(values)
            if force_zero_min:
                y_min = 0.0
            if abs(y_max - y_min) < 1e-6:
                y_max = y_min + 1.0
            margin = 0.12 * (y_max - y_min)
            y_max += margin
            if not force_zero_min:
                y_min -= margin

    # 网格线和Y轴数值
    for i in range(5):
        yy = int(plot_y1 + i * plot_h / 4)
        cv2.line(image, (plot_x1, yy), (plot_x2, yy), GRID_COLOR, 1)
        val = y_max - i * (y_max - y_min) / 4
        put_cv_text(image, f"{val:.0f}", x + 15, yy + 4, scale=0.34, color=TEXT_DIM)

    cv2.line(image, (plot_x1, plot_y1), (plot_x1, plot_y2), (110, 100, 100), 1)
    cv2.line(image, (plot_x1, plot_y2), (plot_x2, plot_y2), (65, 55, 55), 1)

    max_len = 1
    for dq in history_group.values():
        max_len = max(max_len, len(dq))

    if max_len >= 2:
        for key, label, color in series_info:
            arr = list(history_group.get(key, []))
            if len(arr) < 2:
                continue

            prev_pt = None
            for i, value in enumerate(arr):
                if not np.isfinite(value):
                    prev_pt = None
                    continue

                xx = int(plot_x1 + i * (plot_w - 1) / max(1, max_len - 1))
                yy = int(plot_y2 - (float(value) - y_min) / (y_max - y_min) * plot_h)
                yy = max(plot_y1, min(plot_y2, yy))
                curr_pt = (xx, yy)

                if prev_pt is not None:
                    cv2.line(image, prev_pt, curr_pt, color, 2)
                prev_pt = curr_pt

    # 右上角数值，直接压在图上，避免额外拉宽窗口
    legend_x = x + w - 158
    legend_y = y + 46
    legend_h = 20 * len(series_info) + 7
    cv2.rectangle(image, (legend_x - 6, legend_y - 17), (x + w - 10, legend_y + legend_h), (18, 16, 17), -1)
    cv2.rectangle(image, (legend_x - 6, legend_y - 17), (x + w - 10, legend_y + legend_h), (50, 45, 48), 1)

    for i, (key, label, color) in enumerate(series_info):
        v = finite_latest(history_group.get(key, []))
        yy = legend_y + i * 20
        cv2.line(image, (legend_x, yy - 5), (legend_x + 18, yy - 5), color, 2)
        put_cv_text(
            image,
            f"{label}:{fmt_short(v, digits=0)}",
            legend_x + 23,
            yy,
            scale=0.34,
            color=color,
            thickness=1,
        )

    put_cv_text(image, unit_text, x + w - 88, y + 27, scale=0.36, color=TEXT_DIM)


def create_metrics_window(metrics, history, score=None, fps=None):
    """
    独立运动参数窗口：只显示运动参数。
    骨长已经放到 RGB/color 窗口，避免独立窗口顶部表格被挤压显示不全。
    """
    W, H = DASHBOARD_W, DASHBOARD_H
    canvas = np.full((H, W, 3), DASH_BG, dtype=np.uint8)

    # 顶部状态栏
    status = "Motion Parameters | ID0 = highest confidence person"
    if score is not None:
        status += f"   conf:{score:.2f}"
    if fps is not None:
        status += f"   FPS:{fps:.1f}"
    put_cv_text(canvas, status, 14, 24, scale=0.43, color=TEXT_DIM)

    if metrics is None:
        put_cv_text(canvas, "Waiting for valid pose + depth...", 14, 47, scale=0.46, color=TEXT_WARN, thickness=1)

    # 运动参数：波形 + 当前数值
    draw_dark_series_plot(
        canvas,
        10,
        42,
        W - 20,
        205,
        "1. Joint Angle (3D)",
        history["angles"],
        ANGLE_SERIES,
        unit_text="deg",
        fixed_ylim=(0.0, 180.0),
        force_zero_min=True,
    )

    draw_dark_series_plot(
        canvas,
        10,
        257,
        W - 20,
        205,
        "2. Endpoint Velocity (3D)",
        history["linear_velocity"],
        LINEAR_VEL_SERIES,
        unit_text="mm/s",
        fixed_ylim=None,
        force_zero_min=True,
    )

    draw_dark_series_plot(
        canvas,
        10,
        472,
        W - 20,
        238,
        "3. Angular Velocity (3D)",
        history["angular_velocity"],
        ANGULAR_VEL_SERIES,
        unit_text="deg/s",
        fixed_ylim=None,
        force_zero_min=True,
    )

    return canvas

# ============================================================
# 6. 计算骨长、角度、速度
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
    """
    计算三点夹角 ∠ABC，单位：degree。
    p_b 是关节中心点。
    """
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
    theta = np.degrees(np.arccos(cos_theta))

    return float(theta)


def compute_metrics(person_3d, prev_state, curr_time):
    """
    计算：
    1）骨长：肩-肘、肘-腕、髋-膝、膝-踝
    2）关节角度：肩、肘、膝
    3）线速度：手腕、脚踝
    4）角速度：肩、肘、膝

    距离单位：mm
    角度单位：deg
    线速度单位：mm/s
    角速度单位：deg/s
    """
    points = {}
    for name in COCO_KEYPOINT_NAMES:
        points[name] = get_xyz(person_3d, name)

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

    # 肩关节：肘-肩-髋
    # 肘关节：肩-肘-腕
    # 膝关节：髋-膝-踝
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

    metrics = {
        "lengths": lengths,
        "angles": angles,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
    }

    return metrics


# ============================================================
# 7. 选择 ID0：置信度最高的人
# ============================================================
def select_id0_by_highest_confidence(result, keypoints_xy, keypoints_conf):
    """
    当前帧只选择一个人作为 ID0。
    规则：优先使用 YOLO 检测框置信度 result.boxes.conf；
         如果没有 boxes.conf，则使用17个关键点平均置信度。
    返回：selected_index, selected_score, boxes_xyxy
    """
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
            if valid.size > 0:
                scores[i] = float(np.mean(valid))
            else:
                scores[i] = 0.0

    selected_index = int(np.argmax(scores))
    selected_score = float(scores[selected_index])

    return selected_index, selected_score, boxes_xyxy


# ============================================================
# 8. 主函数
# ============================================================
def main():
    # =========================
    # 参数区
    # =========================
    MODEL_PATH = "yolov8n-pose.pt"

    # YOLO推理和显示尺寸
    # RGB显示窗口放大一点；如果电脑卡，可以改回 640, 360
    PROC_W, PROC_H = 800, 450

    KEYPOINT_CONF_THRES = 0.30
    DEPTH_WINDOW = 5

    # 是否显示关键点旁边的3D坐标
    SHOW_3D_COORDS = True

    # True：显示17个点坐标
    # False：只显示肩、肘、腕、髋、膝、踝
    SHOW_ALL_KEYPOINTS = False

    IMPORTANT_KEYPOINTS = {
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    }

    # 上一帧状态，用于计算速度
    prev_state = {
        "time": None,
        "points": {},
        "angles": {}
    }

    # 独立参数窗口的波形历史
    metrics_history = init_motion_history(max_len=120)

    fps_value = None
    fps_last_time = time.time()

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
        print(">>> 当前版本只测 ID0：每帧置信度最高的人作为 ID0")
        print(">>> RGB窗口显示姿态/坐标/骨长；运动参数单独显示在参数窗口，按 q 或 ESC 退出\n")

        cv2.namedWindow("YOLO Pose RGB ID0", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Pose RGB ID0", PROC_W, PROC_H)
        cv2.namedWindow(DASHBOARD_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DASHBOARD_WINDOW_NAME, DASHBOARD_W, DASHBOARD_H)

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

            # 不再用 result.plot() 画全部人，避免多人时混乱
            annotated_rgb = proc_color.copy()

            # 深度图只用于3D坐标/骨长/速度计算，不再单独开深度显示窗口。

            # =========================
            # 只选择 ID0：置信度最高的人
            # =========================
            person_3d = None
            metrics = None
            selected_score = None

            if result.keypoints is not None and result.keypoints.xy is not None:
                keypoints_xy = result.keypoints.xy.cpu().numpy()

                if result.keypoints.conf is not None:
                    keypoints_conf = result.keypoints.conf.cpu().numpy()
                else:
                    keypoints_conf = np.ones(
                        (len(keypoints_xy), len(COCO_KEYPOINT_NAMES)),
                        dtype=np.float32
                    )

                selected_index, selected_score, boxes_xyxy = select_id0_by_highest_confidence(
                    result,
                    keypoints_xy,
                    keypoints_conf
                )

                if selected_index is not None:
                    kp_xy_proc = keypoints_xy[selected_index]
                    kp_conf = keypoints_conf[selected_index]

                    selected_box = None
                    if boxes_xyxy is not None and selected_index < len(boxes_xyxy):
                        selected_box = boxes_xyxy[selected_index]

                    # RGB窗口只画 ID0
                    annotated_rgb = draw_selected_pose(
                        annotated_rgb,
                        kp_xy_proc,
                        kp_conf,
                        box_xyxy=selected_box,
                        score=selected_score,
                        conf_thres=KEYPOINT_CONF_THRES
                    )

                    person_3d = {
                        "id": 0,
                        "keypoints": {}
                    }

                    for kpt_id, name in enumerate(COCO_KEYPOINT_NAMES):
                        x_proc, y_proc = kp_xy_proc[kpt_id]
                        conf = float(kp_conf[kpt_id])

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

                    curr_time = time.time()
                    metrics = compute_metrics(person_3d, prev_state, curr_time)

            # 没检测到人时，清空上一帧状态，避免重新检测后速度突然跳变
            if person_3d is None:
                prev_state["time"] = None
                prev_state["points"] = {}
                prev_state["angles"] = {}

            # =========================
            # RGB窗口：显示姿态、3D坐标、骨长表格
            # =========================
            if SHOW_3D_COORDS:
                annotated_rgb = draw_3d_coords_near_keypoints(
                    annotated_rgb,
                    person_3d,
                    show_all_keypoints=SHOW_ALL_KEYPOINTS,
                    important_keypoints=IMPORTANT_KEYPOINTS
                )

            # 骨长只放在 color/RGB 窗口，运动参数窗口不再显示骨长
            annotated_rgb = draw_bone_length_on_color(
                annotated_rgb,
                metrics,
                score=selected_score
            )

            # =========================
            # 独立运动参数窗口：角度/线速度/角速度，数值 + 波形图
            # =========================
            update_motion_history(metrics_history, metrics)

            now_for_fps = time.time()
            dt_fps = now_for_fps - fps_last_time
            fps_last_time = now_for_fps
            if dt_fps > 1e-6:
                inst_fps = 1.0 / dt_fps
                if fps_value is None:
                    fps_value = inst_fps
                else:
                    fps_value = 0.90 * fps_value + 0.10 * inst_fps

            metrics_window = create_metrics_window(
                metrics,
                metrics_history,
                score=selected_score,
                fps=fps_value
            )

            # =========================
            # 实时显示
            # =========================
            cv2.imshow("YOLO Pose RGB ID0", annotated_rgb)
            cv2.imshow(DASHBOARD_WINDOW_NAME, metrics_window)

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

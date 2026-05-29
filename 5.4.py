import cv2
import numpy as np

# 引入 YOLO
from ultralytics import YOLO

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode, OBFormat, OBPropertyID


def main():
    # ================= 🌟 加载 YOLOv8 Pose 模型 🌟 =================
    print(">>> 正在加载 YOLOv8 Pose 模型...")
    pose_model = YOLO("yolov8n-pose.pt")
    print(">>> 模型加载完成！")
    # ====================================================================

    # 1. 初始化 Pipeline 和 Config
    pipeline = Pipeline()
    config = Config()

    try:
        # 配置深度流
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # 配置彩色流 (尽量请求 RGB888 格式)
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB888, 30)
        except Exception:
            color_profile = color_profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        # 【关键】：开启深度到彩色的硬件对齐 (D2C)，让像素一一对应
        config.set_align_mode(OBAlignMode.HW_MODE)

    except Exception as e:
        print(f"❌ 配置流失败: {e}")
        return

    # 启动相机
    print(">>> 正在连接 Orbbec Femto Mega 相机...")
    pipeline.start(config)

    try:
        # 开启底层帧同步
        pipeline.enable_frame_sync()
        print(">>> 已成功开启彩色与深度帧同步！")
    except Exception as e:
        print(f"❌ 开启同步失败: {e}")
        return

    try:
        while True:
            # 等待获取同步的帧数据
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            # 2. 数据获取与解码
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_matrix = np.reshape(depth_data, (depth_frame.get_height(), depth_frame.get_width()))

            color_data = np.asanyarray(color_frame.get_data())

            if color_frame.get_format() == OBFormat.MJPG:
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            else:
                color_image = np.reshape(color_data, (color_frame.get_height(), color_frame.get_width(), 3))
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # ================= 🌟 核心修改逻辑 🌟 =================

            # 第一步：为了性能，先把 RGB 图像和深度矩阵都缩小到 960x540
            # 采用 INTER_NEAREST 缩放深度图可以避免在边缘产生不存在的插值深度数据
            small_color_image = cv2.resize(color_image, (640, 360))
            small_depth_matrix = cv2.resize(depth_matrix, (640, 360), interpolation=cv2.INTER_NEAREST)

            # 第二步：生成小尺寸的深度伪彩色图 (将作为绘制画布)
            depth_display = cv2.normalize(small_depth_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            small_depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # 第三步：在小尺寸的 RGB 图像上进行 YOLO 推理
            results = pose_model(small_color_image, verbose=False)

            # 第四步：双重绘制 (因为之前开启了 D2C 硬件对齐，所以坐标完全通用)

            # 1. 绘制在彩色图像上 (默认行为)
            annotated_color = results[0].plot()

            # 2. 绘制在深度图像上
            # 传入 img=small_depth_colormap.copy() 作为画布，强制要求 YOLO 在深度伪彩图上画骨架
            annotated_depth = results[0].plot(img=small_depth_colormap.copy())

            # ================================================================

            # 3. 显示画面
            cv2.imshow("YOLOv8 Pose (RGB)", annotated_color)
            cv2.imshow("YOLOv8 Pose (Depth Aligned)", annotated_depth)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n>>> 用户手动中断程序")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
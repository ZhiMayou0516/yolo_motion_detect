import cv2
import numpy as np

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode, OBFormat


def main():
    # 1. 初始化 Pipeline 和 Config
    pipeline = Pipeline()
    config = Config()

    try:
        # 配置深度流
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # 配置彩色流 (尽量请求 RGB888 格式，方便 OpenCV 处理)
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB888, 30)
        except Exception:
            # 如果不支持上述特定格式，则使用默认格式
            color_profile = color_profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        # 【关键】：开启深度到彩色的硬件对齐 (D2C)
        # 这样深度图的尺寸和视角就会变得跟彩色图一模一样
        config.set_align_mode(OBAlignMode.HW_MODE)

    except Exception as e:
        print(f"❌ 配置流失败: {e}")
        return

    # 启动相机
    print(">>> 正在连接 Orbbec Femto Mega 相机...")
    pipeline.start(config)

    # 2. 获取相机内参
    camera_param = pipeline.get_camera_param()

    # ⚠️ 注意：因为做了 D2C 对齐，我们要用 彩色相机 的内参来做 3D 反算！
    color_intrinsic = camera_param.rgb_intrinsic

    fx = color_intrinsic.fx
    fy = color_intrinsic.fy
    cx = color_intrinsic.cx
    cy = color_intrinsic.cy

    # 目标像素点
    target_x = 940
    target_y = 500

    frame_count = 0

    try:
        while True:
            # 等待获取同步的帧数据 (超时时间 100ms)
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            # 处理深度图像数据
            # depth_data = np.asanyarray(depth_frame.get_data())
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_width = depth_frame.get_width()
            depth_height = depth_frame.get_height()
            # 使用 reshape 将一维数组转换为二维矩阵
            depth_matrix = np.reshape(depth_data, (depth_height, depth_width))

            # 处理彩色图像数据 (为了能在 cv2 中显示)
            color_data = np.asanyarray(color_frame.get_data())
            color_width = color_frame.get_width()
            color_height = color_frame.get_height()

            # 根据格式进行转换 (假设获取到的是 RGB888 或类似的可 reshape 格式)
            try:
                color_image = np.reshape(color_data, (color_height, color_width, 3))
                # OpenCV 默认使用 BGR 通道，因此需要将 RGB 转 BGR
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            except ValueError:
                # 如果是 MJPG 格式，需要用 imdecode 解码
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)


            # 将深度图做伪彩色处理，方便人眼观察
            # 深度值通常很大，除以一个系数映射到 0-255，然后转为 uint8
            depth_display = cv2.normalize(depth_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)



            # 3. 用 2 个 cv 窗口实时显示信息
            # 缩放显示窗口，防止 1080p 图像撑爆屏幕
            cv2.imshow("Color Image (Press ESC to exit)", cv2.resize(color_image, (640, 360)))
            cv2.imshow("Depth Image (Aligned)", cv2.resize(depth_colormap, (640, 360)))

            # 按下 'q' 或 'ESC' 键退出
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n>>> 用户手动中断程序")
    finally:
        # 释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print(">>> 相机已安全关闭，窗口已销毁。")


if __name__ == '__main__':
    main()
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode


def print_params(name, intrinsic, distortion):
    """
    按照要求的格式打印参数
    """
    print(f"\n# {name} 核心参数")
    print(f"分辨率: {intrinsic.width}x{intrinsic.height}")
    print(f"焦距 fx: {intrinsic.fx:.4f}")
    print(f"焦距 fy: {intrinsic.fy:.4f}")
    print(f"光心 cx: {intrinsic.cx:.4f}")
    print(f"光心 cy: {intrinsic.cy:.4f}")

    # 提取畸变系数 [k1, k2, p1, p2, k3]
    disto_list = [
        distortion.k1, distortion.k2,
        distortion.p1, distortion.p2,
        distortion.k3
    ]
    print(f"畸变系数 [k1, k2, p1, p2, k3]: {disto_list}")


def main():
    pipeline = Pipeline()
    config = Config()

    try:
        # 2. 配置并启用彩色流
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        # 3. 配置并启用深度流
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # 4. 【关键】：开启硬件层 D2C 对齐
        # 开启后，深度图将通过硬件计算对齐到彩色图分辨率和视角
        config.set_align_mode(OBAlignMode.HW_MODE)

        # 5. 启动 Pipeline
        pipeline.start(config)
        print(">>> 相机已启动并开启硬件 D2C 对齐模式")

        # 6. 获取相机参数
        # 注意：在 D2C 开启状态下，返回的参数是根据对齐配置调整后的
        camera_param = pipeline.get_camera_param()

        # 打印 RGB 参数
        print_params("RGB 相机 (Color)", camera_param.rgb_intrinsic, camera_param.rgb_distortion)

        # 打印 深度 参数
        print_params("深度相机 (Depth)", camera_param.depth_intrinsic, camera_param.depth_distortion)

    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
    finally:
        pipeline.stop()
        print("\n>>> Pipeline 已安全停止")


if __name__ == "__main__":
    main()
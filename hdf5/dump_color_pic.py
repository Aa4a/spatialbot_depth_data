import h5py
import os
import cv2
import numpy as np


def decoder_image(camera_rgb_images):
    if isinstance(camera_rgb_images[0], np.uint8):
        try:
            print(f"尝试解码单张图像，数据长度: {len(camera_rgb_images)}")
            rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
            if rgb is None:
                print("解码单张图像失败，返回 None")
            return rgb
        except Exception as e:
            print(f"解码单张图像时出错: {e}")
            return None
    else:
        rgb_images = []
        for i, camera_rgb_image in enumerate(camera_rgb_images):
            try:
                print(f"尝试解码第 {i} 张图像，数据长度: {len(camera_rgb_image)}")
                rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
                if rgb is None:
                    print(f"解码第 {i} 张图像失败，返回 None")
                else:
                    rgb_images.append(rgb)
            except Exception as e:
                print(f"解码多张图像时第 {i} 张出错: {e}")
        if rgb_images:
            rgb_images = np.asarray(rgb_images)
        return rgb_images


base_path = '/home/ps/nfsroot/DATA/VLM_DATA/LMUData_tagging/data'
for i in os.listdir(base_path):
    base_folder = os.path.join(base_path, i)
    for i in os.listdir(base_folder):
        if i.endswith('hdf5'):
            file_path = os.path.join(base_folder, i)
            output_folder = base_folder

            # 创建保存图片的文件夹（如果不存在）
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            try:
                with h5py.File(file_path, 'r', swmr=True) as root:
                    is_sim = root.attrs['sim']
                    is_compress = root.attrs['compress']
                    print(f"文件 {file_path} 的 is_compress 属性值为: {is_compress}")
                    print(f"文件 {file_path} 的 is_sim 属性值为: {is_sim}")

                    sensor_names = list(root['observations'].keys())
                    for sensor_name in sensor_names:
                        if 'depth' in sensor_name.lower():
                            print(f"跳过深度图: {sensor_name}")
                            continue
                        sensor_group = root['observations'][sensor_name]
                        camera_names = list(sensor_group.keys())
                        for cam_name in camera_names:
                            camera_rgb_images = sensor_group[cam_name][:]
                            # 如果数据未压缩，直接保存（这里假设数据是正确的图像格式）
                            if len(camera_rgb_images.shape) == 3:  # 假设是三维数组表示单张彩色图像
                                print(f"检测到单张未压缩图像，形状: {camera_rgb_images.shape}")
                                image_filename = os.path.join(output_folder,
                                                              f"{sensor_name}_{cam_name}_0.jpg")
                                cv2.imwrite(image_filename, camera_rgb_images)
                                print(f"已保存图片: {image_filename}")
                            elif len(camera_rgb_images.shape) == 4:  # 假设是四维数组表示多张彩色图像
                                print(f"检测到多张未压缩图像，形状: {camera_rgb_images.shape}")
                                for index, img in enumerate(camera_rgb_images):
                                    image_filename = os.path.join(output_folder,
                                                                  f"{sensor_name}_{cam_name}_{index}.jpg")
                                    cv2.imwrite(image_filename, img)
                                    print(f"已保存图片: {image_filename}")
                            else:
                                print(f"检测到压缩图像，尝试解码，形状: {camera_rgb_images.shape}")
                                decode_rgb = decoder_image(camera_rgb_images)
                                print('decode_rgb: ', decode_rgb)
                                if decode_rgb is not None:
                                    if len(decode_rgb.shape) == 3:  # 单张解码后的图像
                                        image_filename = os.path.join(output_folder,
                                                                      f"{sensor_name}_{cam_name}_0.jpg")
                                        cv2.imwrite(image_filename, decode_rgb)
                                        print(f"已保存图片: {image_filename}")
                                    else:
                                        for index, img in enumerate(decode_rgb):
                                            # 生成图片文件名
                                            image_filename = os.path.join(output_folder,
                                                                          f"{sensor_name}_{cam_name}_{index}.jpg")
                                            cv2.imwrite(image_filename, img)
                                            print(f"已保存图片: {image_filename}")
            except Exception as e:
                print(e)
                continue
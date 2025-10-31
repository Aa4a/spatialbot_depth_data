import os
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
from concurrent.futures import ThreadPoolExecutor

# 实例化 CvBridge
bridge = CvBridge()


def extract_color_images(bag_path, output_dir):
    print(f"[INFO] Processing: {bag_path}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with rosbag.Bag(bag_path, "r") as bag:
            topic_name = "/camera/color/image_raw"
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                try:
                    # 使用 cv_bridge 转换为 OpenCV 图像
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    timestamp = t.to_nsec()  # 使用纳秒作为唯一命名
                    filename = f"{os.path.basename(bag_path).replace('.bag', '')}_{timestamp}.png"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, img)
                    print(f"[SAVED] {filepath}")
                except Exception as e:
                    print(f"[ERROR] Failed to process image at time {t}: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to open bag file {bag_path}: {e}")


def process_bag_files(base_dir):
    print(f"[INFO] Starting batch processing in: {base_dir}")
    with ThreadPoolExecutor() as executor:
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for file in os.listdir(subdir_path):
                if file.endswith(".bag"):
                    bag_path = os.path.join(subdir_path, file)
                    executor.submit(extract_color_images, bag_path, subdir_path)


if __name__ == "__main__":
    base_dir = '/mnt/nfs/DATA/VLM_DATA/LMUData_tagging/data'
    process_bag_files(base_dir)

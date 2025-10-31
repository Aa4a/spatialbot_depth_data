import os
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    timestamp = t.to_nsec()
                    filename = f"{os.path.basename(bag_path).replace('.bag', '')}_{timestamp}.png"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, img)
                    print(f"[SAVED] {filepath}")
                except Exception as e:
                    print(f"[ERROR] Failed to process image at time {t}: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to open bag file {bag_path}: {e}")

if __name__ == "__main__":
    bag_files = []
    root_dir = "/mnt/nfsroot/DATA/VLM_DATA/LMUData_tagging/data"

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(dirpath, file))

    print(f"Found {len(bag_files)} .bag files")

    # 设置线程池数量（可根据 CPU 核心数和 I/O 负载调整）
    max_workers = min(8, len(bag_files))  # 或 os.cpu_count()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for bag_path in bag_files:
            output_dir = os.path.dirname(bag_path)
            futures.append(executor.submit(extract_color_images, bag_path, output_dir))

        for future in as_completed(futures):
            # 捕获可能的异常
            try:
                future.result()
            except Exception as e:
                print(f"[THREAD ERROR] {e}")


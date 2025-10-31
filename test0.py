import rosbag
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor


# 读取 bag 并提取彩色图像
def extract_color_images(bag_path, output_dir):
    print(f"Processing: {bag_path}")

    bag = rosbag.Bag(bag_path, "r")
    topic_name = "/camera/color/image_raw"

    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        # 转换数据
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        img = img_array.reshape((msg.height, msg.width, -1))  # (H, W, C)

        # 生成保存路径
        img_filename = os.path.join(output_dir, f"{os.path.basename(bag_path)}_{t}.png").replace('.bag', '').replace('-', '_')
        cv2.imwrite(img_filename, img)
        print(f"Saved: {img_filename}")

    bag.close()


# def extract_all_images_from_bag(bag_path, output_dir):
#     print(f"Processing: {bag_path}")
#     bag = rosbag.Bag(bag_path, "r")
#
#     os.makedirs(output_dir, exist_ok=True)
#     topic_types = bag.get_type_and_topic_info()[1]
#
#     image_topics = [topic for topic, info in topic_types.items() if info.msg_type == 'sensor_msgs/Image']
#     print(f"Found image topics: {image_topics}")
#
#     image_count = {}
#
#     for topic, msg, t in bag.read_messages(topics=image_topics):
#         try:
#             # 获取图像格式
#             if msg.encoding in ['rgb8', 'bgr8']:
#                 dtype = np.uint8
#                 channels = 3
#             elif msg.encoding in ['mono8']:
#                 dtype = np.uint8
#                 channels = 1
#             elif msg.encoding in ['mono16', '16UC1']:
#                 dtype = np.uint16
#                 channels = 1
#             else:
#                 print(f"Unsupported encoding: {msg.encoding} at topic {topic}")
#                 continue
#
#             img_array = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width, channels)) \
#                         if channels > 1 else np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width))
#
#             # 构造保存路径
#             safe_topic = topic.replace('/', '_')
#             timestamp_str = str(t.to_nsec())
#             filename = os.path.join(output_dir, f"{safe_topic}_{timestamp_str}.png")
#             cv2.imwrite(filename, img_array)
#             image_count[topic] = image_count.get(topic, 0) + 1
#             print(f"Saved image: {filename}")
#         except Exception as e:
#             print(f"Error processing image from {topic} at time {t}: {e}")
#
#     bag.close()
#     print("Done. Image counts per topic:")
#     for topic, count in image_count.items():
#         print(f"{topic}: {count} images")



def process_bag_files(base_dir):
    # 使用 ThreadPoolExecutor 进行多线程处理
    with ThreadPoolExecutor() as executor:
        for subdir in os.listdir(base_dir):
            base_path = os.path.join(base_dir, subdir)
            for file in os.listdir(base_path):
                if file.endswith(".bag"):
                    bag_path = os.path.join(base_path, file)
                    executor.submit(extract_color_images, bag_path, base_path)


if __name__ == "__main__":
    base_dir = '/mnt/nfs/DATA/VLM_DATA/LMUData_tagging/data'
    process_bag_files(base_dir)

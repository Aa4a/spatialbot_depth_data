import os
import shutil
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Read the JSON file for list7
with open("list7.json", 'r', encoding="utf-8") as f:
    list7 = json.load(f)

# Folder list
list1 = [
    "03-29-test", "1212", "123", "2025-02-17", "2025-02-18", "2025-02-19", "2025-02-20", "2025-02-21",
    "2025-02-22", "2025-02-23", "2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28",
    "2025-03-", "2025-03-01", "2025-03-03", "2025-03-05", "2025-03-05.1", "2025-03-06", "2025-03-07",
    "2025-03-08", "2025-03-09", "2025-03-10", "2025-03-11", "2025-03-12", "2025-03-12a", "2025-03-13",
    "2025-03-14", "2025-03-14.1", "2025-03-15", "2025-03-17", "2025-03-18", "2025-03-19", "2025-03-20",
    "2025-03-21_new", "2025-03-23", "2025-03-24", "2025-03-24_new", "2025-03-25", "2025-03-26",
    "2025-03-27", "2025-03-28", "2025-03-29-test", "2025-03-30", "2025-03-31", "2025-3-31",
    "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-05", "2025-04-06", "2025-04-07", "2025-04-08",
    "2025-04-10", "250407", "faranka_02-test", "faranka_02_test_apple", "franka", "Franka_04_test",
    "franka_dual_0401", "test123", "UR_TEST", "ur_test_29"
]

base_path = "/mnt/nas_nfs"
target_dir = "/home/ps/nfsroot/DATA/VLM_DATA/LMUData_tagging/data"

# Collect all unprocessed paths
task_list = []
for i in list1:
    path = os.path.join(base_path, i)
    if not os.path.exists(path):
        continue
    for j in os.listdir(path):
        path2 = os.path.join(path, j)
        if not os.path.isdir(path2):
            continue
        for k in os.listdir(path2):
            path3 = os.path.join(path2, k)
            if not os.path.isdir(path3):
                continue
            for l in os.listdir(path3):
                if l not in list7:
                    task_path = os.path.join(path3, l)
                    task_list.append(task_path)

# Define copy function (for multithreading)
def copy_task_data(task_path):
    succeed_path = os.path.join(task_path, "success_episodes")
    if not os.path.exists(succeed_path):
        return

    for j in os.listdir(succeed_path):
        data_path = os.path.join(succeed_path, j, "data")
        if not os.path.exists(data_path):
            continue

        files = os.listdir(data_path)
        if len(files) == 1 and files[0] == "data.rrd":
            continue

        # Target folder name (last directory level)
        dst_folder_name = Path(task_path).name
        dst_path = os.path.join(target_dir, dst_folder_name)
        os.makedirs(dst_path, exist_ok=True)

        for file_name in files:
            src_file = os.path.join(data_path, file_name)
            dst_file = os.path.join(dst_path, file_name)
            try:
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"复制文件: {src_file} -> {dst_file}")
            except Exception as e:
                print(f"复制出错：{src_file} -> {dst_file}, 错误: {e}")

# Multithreaded task execution
max_workers = min(16, os.cpu_count() or 4)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(copy_task_data, path) for path in task_list]
    for future in as_completed(futures):
        pass  # Optionally, you can add progress logging here

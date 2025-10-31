import os
import shutil

# 指定文件夹路径
source_dir = '/home/ps/nfsroot/DATA/VLM_DATA/LMUData_tagging/data'
destination_dir = '.'  # 当前目录

# 遍历直接子文件夹
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    print(folder_path)
    for file in os.listdir(folder_path):
        print(file)
        if file.endswith('.bag'):
            # 找到第一个 .bag 文件，复制到当前目录
            source_file = os.path.join(folder_path, file)
            shutil.copy(source_file, destination_dir)
            print(f"文件 {file} 已复制到当前目录.")
            break  # 只复制第一个找到的 .bag 文件虑文件夹


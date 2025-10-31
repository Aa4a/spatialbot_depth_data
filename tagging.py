import os
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from PIL import Image, ImageTk
import json
import numpy as np
from openai import OpenAI
import platform


openai_api_key = "EMPTY"
openai_api_base = "http://180.76.145.157:8888/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def convert_path(linux_path):
    if os.name == "posix" or platform.system() == "Linux":
        return linux_path
    elif os.name == "nt" or platform.system() == "Windows":
        path = linux_path.replace("/nfsroot/DATA/VLM_DATA", "Z:\\DATA\\VLM_DATA").replace("/", "\\")
        return path
    return linux_path


def qwen(eng_text):
    if eng_text.isdigit():
        return eng_text
    completion = client.chat.completions.create(
        model="Qwen2_5-72B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": f'\n将上面英文翻译成中文，只返回翻译后的中文，不用有其他内容\n{eng_text}'}]}
        ]
    )
    return completion.choices[0].message.content


data_name = filedialog.askopenfilename(title="选择一个 JSON 文件", filetypes=[("JSON 文件", "*.json")])
# data_name = r"Z:\DATA\VLM_DATA\LMUData_tagging\spatialbot\SpatialQA\SpatialQA_1000.json"
save_name = data_name

with open(data_name, 'r', encoding="utf-8") as file:
    data = json.load(file)

max_index = len(data)
current_index = 0

root = tk.Tk()
root.title("Depth Measurement Quiz")
option_var = tk.StringVar()


def normalize_depth_image(image_path, target_size=(400, 400)):
    try:
        image_path = convert_path(image_path)
        img_depth = Image.open(image_path)
        if img_depth.mode == 'I;16':
            img_depth = img_depth.point(lambda i: i * (255.0 / 65535.0)).convert("L")
        elif img_depth.mode == 'L':
            img_depth = img_depth.convert("L")
        img_depth_np = np.array(img_depth)
        img_depth_np = (img_depth_np - np.min(img_depth_np)) / (np.max(img_depth_np) - np.min(img_depth_np)) * 255
        img_depth_normalized = Image.fromarray(img_depth_np.astype(np.uint8)).resize(target_size,
                                                                                     Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img_depth_normalized)
    except Exception as e:
        print(f"加载和归一化深度图像 {image_path} 时出错: {e}")
        return None


def display_data(data, index=0):
    global current_index
    current_index = index
    for frame in root.winfo_children():
        frame.destroy()

    image_frame = tk.Frame(root)
    pictures = data[index]["image"]
    folder_path = os.path.dirname(data_name)

    # 处理 RGB 图片路径
    rgb_picture = pictures[0]
    if not os.path.isabs(rgb_picture):
        rgb_picture = os.path.join(folder_path, rgb_picture)
    rgb_picture = convert_path(rgb_picture)

    try:
        img_rgb = Image.open(rgb_picture).resize((400, 400), Image.Resampling.LANCZOS)
        img_rgb = ImageTk.PhotoImage(img_rgb)
    except Exception as e:
        print(f"加载 RGB 图片 {rgb_picture} 时出错: {e}")
        img_rgb = None

    panel_rgb = tk.Label(image_frame, image=img_rgb)
    panel_rgb.image = img_rgb
    panel_rgb.pack(side="left")

    # 当有深度图片时才加载并显示
    if len(pictures) > 1:
        depth_picture = pictures[1]
        if not os.path.isabs(depth_picture):
            depth_picture = os.path.join(folder_path, depth_picture)
        depth_picture = convert_path(depth_picture)
        img_depth = normalize_depth_image(depth_picture)
        if img_depth:
            panel_depth = tk.Label(image_frame, image=img_depth)
            panel_depth.image = img_depth
            panel_depth.pack(side="left")

    image_frame.pack(pady=10)

    conversation_frame = tk.Frame(root)
    conversation_frame.pack(fill="both", expand=True)
    text_widget = tk.Text(conversation_frame, wrap="word", height=10, font=("Arial", 12))
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(conversation_frame, orient="vertical", command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    text_widget.config(yscrollcommand=scrollbar.set)

    all_text = ""
    for msg in data[index]['conversations']:
        role = "Question" if msg['from'] == "human" else "Answer"
        text_widget.insert("end", f"{role}: {msg['value']}\n")
        all_text += f"{role}: {msg['value']}\n"

    translation = qwen(all_text)
    text_widget.insert("end", f"翻译：{translation}")
    text_widget.config(state="disabled")

    next_button = tk.Button(root, text="下一题", command=lambda: switch(data, current_index + 1))
    next_button.pack(pady=5)

    index_frame = tk.Frame(root)
    index_frame.pack(pady=5)
    index_label = tk.Label(index_frame, text=f"{current_index + 1}/{max_index}")
    index_label.pack(side="left")
    index_entry = tk.Entry(index_frame, width=5)
    index_entry.pack(side="left", padx=5)
    jump_button = tk.Button(index_frame, text="跳转", command=lambda: jump_to_index(data, index_entry.get()))
    jump_button.pack(side="left")

    option = data[index].get("option", "")
    option_var.set(option)
    option_frame = tk.Frame(root)
    option_frame.pack(pady=5)
    right_button = tk.Radiobutton(option_frame, text="right", variable=option_var, value="right",
                                  command=lambda: change_current_option(data, current_index, "right"))
    right_button.pack(side="left", padx=10)
    mistake_button = tk.Radiobutton(option_frame, text="mistake", variable=option_var, value="mistake",
                                    command=lambda: change_current_option(data, current_index, "mistake"))
    mistake_button.pack(side="left", padx=10)
    save_button = tk.Button(root, text="保存", command=lambda: save_data(data))
    save_button.pack(pady=10)


def switch(data, index):
    if index >= max_index:
        save_data(data)
        root.destroy()
    else:
        display_data(data, index)


def jump_to_index(data, index_str):
    try:
        index = int(index_str) - 1
        if 0 <= index < max_index:
            display_data(data, index)
        else:
            messagebox.showerror("错误", "输入的索引超出范围")
    except ValueError:
        messagebox.showerror("错误", "请输入有效的整数索引")


def change_current_option(data, index, option):
    data[index]["option"] = option


def save_data(data):
    with open(save_name, 'w') as file:
        json.dump(data, file, indent=4)
        print(f'saved in {save_name}')


if __name__ == '__main__':
    display_data(data)
    root.mainloop()
import copy
from prompt_lists import depth_value_prompts, spatial_understanding_prompts, scene_understanding_prompts
import json
import base64
import os
import re
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers
import warnings
import random
import concurrent.futures
from openai import OpenAI
import openai


# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = 'cuda'  # or cpu

# model_name = 'RussRobin/SpatialBot-3B'
model_name = 'SpatialBot-3B'
offset_bos = 0

# create model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)


openai_api_key = "EMPTY"
openai_api_base = "http://qwen25-vlm.x-humanoid-cloud.com:8888/v1"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



def GPT(messages, client):
    try:
        completion = client.chat.completions.create(
            model="Qwen2_5-VL-72B-Instruct",
            messages=messages
        )
        response = completion.choices[0].message.content
        return response

    except openai.OpenAIError as e:  # 捕获 OpenAI 相关错误
        print(f"OpenAI API Error: {e}")
        return None
    except Exception as e:  # 捕获其他未知错误
        print(f"Unexpected Error: {e}")
        return None


def depth_api(image, depth_image, questions):
    print(questions)
    length = len(questions)
    image1 = Image.open(image)
    image2 = Image.open(depth_image)

    channels = len(image2.getbands())
    if channels == 1:
        # Convert the image to a NumPy array with a larger data type (e.g., uint16)
        img = np.array(image2, dtype=np.uint16)  # Convert to uint16 to avoid overflow
        height, width = img.shape

        # Perform calculations using the larger data type
        three_channel_array = np.zeros((height, width, 3), dtype=np.uint16)
        three_channel_array[:, :, 0] = (img // 1024) * 4
        three_channel_array[:, :, 1] = (img // 32) * 8
        three_channel_array[:, :, 2] = (img % 32) * 8

        # Clip values to the range [0, 255] and convert to uint8
        three_channel_array = np.clip(three_channel_array, 0, 255).astype(np.uint8)

        image2 = Image.fromarray(three_channel_array, 'RGB')

    image_tensor = model.process_images([image1, image2], model.config).to(dtype=model.dtype, device=device)

    # If 'Expected all tensors to be on the same device' error is thrown, uncomment the following line
    model.get_vision_tower().to('cuda')

    response = ""
    for index, question in enumerate(questions):
        prompt = f"What is the depth value of {question}? Answer directly from depth map."
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:],
                                 dtype=torch.long).unsqueeze(0).to(device)
        # generate
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=100,
            use_cache=True,
            repetition_penalty=1.0  # increase this to avoid chattering
        )[0]

        result = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        if index < length - 1:
            response += f"{result.replace('.', '')}, "
        else:
            response += f"{result}"
    print("response-----------------", response)
    return response


text1 = '''
        Design a conversation between you and a person discussing depth maps.
        When the human asks you to describe a depth map, you should focus on the prediction of depth values.
        Colors only represent depth values. Do not directly mention the colors in your answer, but refer to the depth distribution they represent.
        When reviewing the depth map, you should also infer what might be present in the image.
        If certain objects are present in the RGB image and can be inferred from the depth map, you may mention them.
        Pay attention to spatial relationships when possible.
        When referring to spatial relationships, such as left and right, use the real-world left and right, not the left and right of the image coordinate system.
        '''

text2 = '''
        Design a conversation with no more than 3 question-answer pairs, discussing the image.
        The conversation should be logically coherent.
        Consider the spatial relationships of objects in the image.
        When describing spatial relationships, always use real-world directions as if you are standing in the actual scene.
        For example, when referring to the right side of an object, it should mean the real-world right side of the object, not the right side of the image.
        Only describe what you are certain about.
        '''

text3 = '''
        Design a dialogue with up to three question-and-answer pairs, with the content logically connected to the image.
        First, think about what task the robot might be performing.
        Then, generate the dialogue based on the robot's task, describing only the confirmed information.
        Please note, you are discussing the image and the robot with a person, not the robot itself.
        '''

system = {"role": "system", "content": "You are a native English speaker."}


def regular(response):
    # 使用正则表达式提取 JSON 部分
    json_pattern = re.compile(r'\[.*\]', re.DOTALL)
    json_match = json_pattern.search(response)

    if json_match:
        json_str = json_match.group(0)
        # 尝试解析 JSON
        try:
            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as e:
            return None
    else:
        return None


def process(image, client):
    if image.endswith(".png"):
            depth_image = image.replace('.png', '_depth.png')
    elif image.endswith(".jpg"):
        depth_image = image.replace('.jpg', '_depth.jpg')

    json_list = []
    id = 0
    for s in range(0, 25):
        random1 = random.randint(0, 1)
        dep = depth_value_prompts[random1]
        prompt1 = f'''
                {text1}

                Question type:
                {dep}

                Finally, you need to add the objects involved in the problem to the list and return them to me, for example:
                ["xxx", "yyy", "zzz"]

                Return format：
                [{{"role": "user", "content": [{{"type": "text", "text": "......(The question you raised)"}}]}},
                    {{"role": "assistant", "content": [{{"type": "text", "text": "......(Your Answer)"}}]}},
                    ["xxx", "yyy", "zzz"]
                    ......
                    ]
                    (The last element in the list is the list of objects involved in the problem)
                '''
        rgb_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}}
        depth_content = {"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(depth_image)}"}}
        messages = [
            system,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt1},
                    rgb_content,
                    depth_content,
                ],
            },
        ]
        try:
            response = GPT(messages, client)
        except Exception as e:
            print("Error", e)
            continue

        parsed_data = regular(response)

        if parsed_data is None:
            continue
        objects = parsed_data[-1]
        regular_prompt = parsed_data[:-1]

        regular_prompt[0]["content"].append(rgb_content)
        regular_prompt[0]["content"].append(depth_content)
        depths = depth_api(image, depth_image, objects)
        regular_prompt[0]["content"][0]['text'] += f", Depth: {depths}"

        random2 = random.randint(0, 5)
        spatial = spatial_understanding_prompts[random2]

        prompt2 = f'''
                    {text2}

                    Question type:
                    {spatial}

                    Return format：
                    [{{"role": "user", "content": [{{"type": "text", "text": "......(The question you raised)"}}]}},
                    {{"role": "assistant", "content": [{{"type": "text", "text": "......(Your Answer)"}}]}}
                    ......
                    ]
                    '''

        messages = [
            system,
        ]
        for i in regular_prompt:
            messages.append(i)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt2}
            ],
        })
        try:
            response = GPT(messages, client)
        except Exception as e:
            print("Error", e)
            continue
        regular_prompt2 = regular(response)
        if regular_prompt2 is None:
            continue

        random3 = random.randint(0, 5)
        scene = scene_understanding_prompts[random3]
        prompt3 = f'''
                    {text3}

                    Question type:
                    {scene}

                    Return format：
                    [{{"role": "user", "content": [{{"type": "text", "text": "......(The question you raised)"}}]}},
                    {{"role": "assistant", "content": [{{"type": "text", "text": "......(Your Answer)"}}]}}
                    ......
                    ]
                    '''
        messages = [
            system
        ]
        for i in regular_prompt:
            messages.append(i)
        for j in regular_prompt2:
            messages.append(j)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt3}
            ],
        })
        try:
            response = GPT(messages, client)
        except:
            continue
        regular_prompt3 = regular(response)
        for k in regular_prompt3:
            messages.append(k)

        history = regular_prompt + regular_prompt2 + regular_prompt3
        result = copy.deepcopy(history)
        del result[0]['content'][1]
        del result[0]['content'][1]
        id += 1
        json1 = {
            "id": id,
            "image": [image, depth_image],
            "conversations": result
        }
        print(json1)
        json_list.append(json1)
    return json_list




def process_image(image, client):
    return process(image, client)

def even_sample(lst, num_samples):
    if len(lst) < num_samples:
        raise ValueError(f"列表长度不足 {num_samples}")
    step = len(lst) / num_samples
    return [lst[int(i * step)] for i in range(num_samples)]


def save_json(json_list, file_count):
    """保存 JSON 数据到文件，每 50,000 条保存一个文件"""
    filename = f"result_{file_count}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)
    print(f"Saved {filename} with {len(json_list)} records.")



if __name__ == '__main__':
    from pathlib import Path

    with open("result_0407.json", 'r', encoding="utf-8") as f7:
        data7 = json.load(f7)
    with open("result_0408.json", 'r', encoding="utf-8") as f8:
        data8 = json.load(f8)
    with open("result_0414.json", 'r', encoding="utf-8") as f9:
        data9 = json.load(f9)
    data10 = data7 + data8 + data9
    list7 = [Path(i["image"][0]).parent.name for i in data10]

    print(list7)
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    base_path1 = "/home/ps/nfsroot/DATA/VLM_DATA/LMUData_tagging/data"
    base_path2 = "/home/ps/nfsroot/DATA/VLM_DATA/LMUData_tagging/spatialbot/data"
    base_path3 = "/home/ps/nfsroot/DATA/VLM_DATA/LMUData_tagging/Robotmind"
    print('start')
    folders1 = [os.path.join(base_path1, i) for i in os.listdir(base_path1) if i not in list7]
    folders2 = [os.path.join(base_path2, i) for i in os.listdir(base_path2) if i not in list7]
    folders3 = [os.path.join(base_path3, i) for i in os.listdir(base_path3) if i not in list7]
    folders = folders1 + folders2 + folders3
    # folders = folders1
    print("folders: ", len(folders))
    image_list = []

    for folder in folders:
        pictures = [os.path.join(folder, i) for i in os.listdir(folder) if 'depth' not in i]
        filtered_pictures = [pic for pic in pictures if pic.endswith(("_top.png", "_top.jpg"))]
        if len(filtered_pictures) < 5:
            filtered_pictures = [pic for pic in pictures if pic.endswith(('.png', '.jpg'))]

        total_images = len(filtered_pictures)
        if total_images >= 10:
            step = total_images // 10
            selected_pictures = [filtered_pictures[i * step] for i in range(10)]
            image_list.extend(selected_pictures)
        else:
            image_list.extend(filtered_pictures)

    # task_list = even_sample(image_list, 1000)
    task_list = []
    for i in image_list:
        if os.path.exists(i):
            print(i)
            task_list.append(i)
    print(task_list)
    print(len(task_list))

    json_list = []
    file_count = 1  # 记录 JSON 文件索引

    # 使用 ThreadPoolExecutor 来并发处理图片
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_image, image, client) for image in task_list]

        for future in concurrent.futures.as_completed(futures):
            try:
                result_list = future.result()
                json_list.extend(result_list)

                # 每 50,000 条数据存一次文件
                if len(json_list) >= 50000:
                    save_json(json_list, file_count)
                    json_list = []  # 清空数据列表
                    file_count += 1  # 更新文件编号

            except Exception as e:
                print(f"An error occurred: {e}")

    # 处理剩余的数据
    if json_list:
        save_json(json_list, file_count)






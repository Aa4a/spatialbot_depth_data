import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageTk
import warnings
import numpy as np

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = 'cuda'  # or cpu

# model_name = 'RussRobin/SpatialBot-3B'
model_name = '/home/ps/python_projects/SpatialBot/SpatialBot-3B'
offset_bos = 0

# create model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)

# text prompt
prompt = 'What is the depth value of Machinery? Answer directly from depth map.'
text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(device)



image1 = Image.open('./test.png')
image2 = Image.open('./test1.png')

channels = len(image2.getbands())
if channels == 1:
    img = np.array(image2, dtype=np.uint16)
    height, width = img.shape
    three_channel_array = np.zeros((height, width, 3), dtype=np.uint16)
    three_channel_array[:, :, 0] = (img // 1024) * 4
    three_channel_array[:, :, 1] = (img // 32) * 8
    three_channel_array[:, :, 2] = (img % 32) * 8

    # Clip values to the range [0, 255] and convert to uint8
    three_channel_array = np.clip(three_channel_array, 0, 255).astype(np.uint8)

    image2 = Image.fromarray(three_channel_array, 'RGB')

image_tensor = model.process_images([image1,image2], model.config).to(dtype=model.dtype, device=device)

# If 'Expected all tensors to be on the same device' error is thrown, uncomment the following line
model.get_vision_tower().to('cuda')

# generate
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    max_new_tokens=100,
    use_cache=True,
    repetition_penalty=1.0 # increase this to avoid chattering
)[0]

print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
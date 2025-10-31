import pandas as pd
import torch

# 替换 'your_file.parquet' 为实际的文件路径
df = pd.read_parquet("/home/ps/convert/241223_upright_cup/data/chunk-000/episode_000095.parquet")
data_array = df.to_numpy()

# 如果需要转换成指定的数据类型，比如浮点型，可加入 dtype 参数
# data_array = df.to_numpy(dtype='float32')

# 将 NumPy 数组转换为 PyTorch 的 tensor
tensor_data = torch.tensor(data_array)

print(tensor_data)
import json

# with open("data.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # for item in data:
# #     item["image"][0] = item["image"][0].replace("/home/ps/nfsroot", "/nfsroot")
# #     item["image"][1] = item["image"][1].replace("/home/ps/nfsroot", "/nfsroot")
#
# # with open("data1.json", "w", encoding="utf-8") as f:
# #     f.write(json.dumps(data, indent=4, ensure_ascii=False))


with open("data1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

N = len(data)
step = N / 500  # 计算步长
subset = [data[int(i * step)] for i in range(500)]  # 均匀取样
with open("data_328.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(subset, indent=4, ensure_ascii=False))
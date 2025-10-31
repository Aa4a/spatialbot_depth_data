import json

mapping = {"user": "human", "assistant": "gpt"}

with open("result_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)

list1 = []

id = 0
for line in data:
    id += 1
    try:
        con_list = []
        for i in line["conversations"]:
            json0 = {
                "from": mapping[i["role"]],
                "value": i["content"][0]["text"]
            }
            con_list.append(json0)
        json1 = {
            "id": id,
            "image": line["image"],
            "conversations": con_list
        }
        list1.append(json1)
    except:
        continue



with open("data_all.json", "w", encoding="utf-8") as f:
    json.dump(list1, f, indent=4)

import json

import os
import sys
d = sys.argv[1]

with open(os.path.join(d, "flame_params.json"), "r") as f1:
    data1 = json.load(f1)["frames"]

with open(os.path.join(d, "facs_params.json"), "r") as f2:
    data2 = json.load(f2)["frames"]

# print(data1[0].keys())

merged_data = {}

# n=0
for item2 in data2:
    file_path = item2["file_path"]
    merged_data[file_path] = {"file_path": file_path, "facs": item2["expression"]}
    # n += 1
    # if n > 2:
    #     break

for item1 in data1:
    file_path = item1["file_path"]
    if file_path in merged_data:
        for key in item1.keys():
            if key != "file_path":
                merged_data[file_path][key] = item1[key]
        # print(merged_data[file_path].keys())
        # exit()
        
    # else:
    #     merged_data[file_path] = {"file_path": file_path, "facs": item2["facs"]}    

merged_list = list(merged_data.values())        

data_to_save = {}
data_to_save["frames"] = merged_list

with open(os.path.join(d, "flame_params.json"), "r") as f1:
    data1 = json.load(f1)

for key in data1.keys():
    if key != "frames":
        data_to_save[key] = data1[key]
print(data_to_save.keys())

print(len(merged_list))
with open(os.path.join(d, "merged_params.json"), "w") as f:
    json.dump(data_to_save, f)
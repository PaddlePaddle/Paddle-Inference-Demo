import json

check_diff_mark_tensor_names = []

def get_mark_names_from_serialized_json():
    with open("cache/engine_info_*.json", "r") as f:
        data = json.load(f)
    for layer in data:
        for output in layer['Outputs']:
            output_name = output["Name"]
            if "subgraph" in output_name:
                output_name = output_name[0:output_name.index("_subgraph")]
                check_diff_mark_tensor_names.append(output_name)

def assign_mark_names():
    flag = False
    start = ["tensor_name.tmp_0", "tensor_name.tmp_1"]
    end = ["tensor_name.tmp_5"]
    with open("save_baseline.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        baseline_tensor_names = line.split(":")[-1].split(" ")[-1].strip()
        if baseline_tensor_names in start:
            flag = True
        if flag:
            # print(baseline_tensor_names)
            check_diff_mark_tensor_names.append(baseline_tensor_names)
        if baseline_tensor_names in end:
            flag = False


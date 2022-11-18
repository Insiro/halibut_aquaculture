import os
from os import path
import json
import random
from shutil import copy2

# ID = 3
pathList = [
    {
        "path": "11.넙치치어_yolo",
        "img_path": "11.넙치치어-QS",
        "name_kr": "넙치치어",
        "name": "halibut",
    }
]
IMAGE_PATH = "./1_원천데이터"
LABEL_PATH = "./2_라벨링데이터"

YOLO_LABEL_PATH = "./labels"
YOLO_IMAGE_PATH = "./images"


def transform2yolo(label):
    annot = label["Annotations"]
    resol = json.loads(label["Info"]["resolution"])
    lines = []
    for an in annot:
        bbox = json.loads(an["rect.points"])
        x = (bbox[0] * 2 + bbox[2]) / (2 * resol[0])
        y = (bbox[1] * 2 + bbox[3]) / (2 * resol[1])
        c_w = bbox[2] / resol[0]
        c_h = bbox[3] / resol[1]
        lines.append(f"{3} {x} {y} {c_w} {c_h}")

    return lines


def change_category_id(file):
    outstr = file.readlines()
    for i, line in enumerate(outstr):
        if len(line) > 0:
            words = line.split(" ")
            words[0] = "3"
            outstr[i] = " ".join(words)
    return "".join(outstr)


data_size = []
total = 0
for label_id, path_info in enumerate(pathList):
    size = 0
    for file_name in os.listdir(path.join(LABEL_PATH, path_info["path"])):
        current = path.join(LABEL_PATH, path_info["path"], file_name)
        outstr = ""
        print(current)
        with open(current) as labeled:
            outstr = change_category_id(labeled)

        rand = random.randrange(0, 10)
        div = "val"
        size += 1
        label_file_path = os.path.join(YOLO_LABEL_PATH, div, file_name)

        with open(label_file_path, "w") as outfile:
            print(outstr, file=outfile, end="")

        file_names = file_name.split(".")
        file_names[-1] = "jpg"
        img_name = (".").join(file_names)

        current_img = path.join(IMAGE_PATH, path_info["img_path"], img_name)

        img_save_path = path.join(YOLO_IMAGE_PATH, div, img_name)
        # print(f"{current_img}\t{img_save_path}")
        copy2(current_img, img_save_path)
    data_size.append(
        {
            "class": 3 + label_id,
            "name": path_info["name"],
            "name_kr": path_info["name_kr"],
            "val": size,
        }
    )
    total += size

info = {}
with open("dataset_information.json", "r", encoding="UTF-8") as file:
    info = json.load(file)
    info["info"]["val"] += total
    info["data_size"].extend(data_size)
with open("dataset_information.json", "w", encoding="UTF-8") as file:
    print(info)
    json.dump(info, file, ensure_ascii=False, indent=4)
    pass

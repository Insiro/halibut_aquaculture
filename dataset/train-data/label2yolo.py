import os
from os import path
import glob
import json
import random
from shutil import copy2

pathList = [
    {
        "path": "11.넙치유생-QL/LV10",
        "img_path": "11.넙치유생-QS/LV10",
        "name_kr": "넙치유생_변태1단계",
        "name": "halibut_larva_step1",
    },
    {
        "path": "11.넙치유생-QL/LV20",
        "img_path": "11.넙치유생-QS/LV20",
        "name_kr": "넙치유생_변태2단계",
        "name": "halibut_larva_step2",
    },
    {
        "path": "11.넙치유생-QL/LV30",
        "img_path": "11.넙치유생-QS/LV30",
        "name_kr": "넙치유생_변태3단계",
        "name": "halibut_larva_step3",
    },
    # {"path": "11.넙치치어-QL", "name_kr": "넙치치어", "name": "halibut"},
]
IMAGE_PATH = "./1_원천데이터"
LABEL_PATH = "./2_라벨링데이터"

YOLO_LABEL_PATH = "./labels"
YOLO_IMAGE_PATH = "./images"

# generate folders
for p in [YOLO_IMAGE_PATH, YOLO_LABEL_PATH]:
    if not path.isdir(p):
        os.mkdir(p)
    for div in ["train", "test", "val"]:
        ddiv = path.join(p, div)
        if not path.isdir(ddiv):
            os.mkdir(ddiv)
        else:
            for f in glob.glob(ddiv + "/*"):
                os.remove(f)


def transform2yolo(label_id: int, label):
    annot = label["Annotations"]
    resol = json.loads(label["Info"]["resolution"])
    lines = []
    for an in annot:
        bbox = json.loads(an["rect.points"])
        x = (bbox[0] * 2 + bbox[2]) / (2 * resol[0])
        y = (bbox[1] * 2 + bbox[3]) / (2 * resol[1])
        c_w = bbox[2] / resol[0]
        c_h = bbox[3] / resol[1]
        lines.append(f"{label_id} {x} {y} {c_w} {c_h}")

    return lines


data_size = []
total = [0, 0]
for label_id, path_info in enumerate(pathList):
    test = 0
    train = 0
    for file_name in os.listdir(path.join(LABEL_PATH, path_info["path"])):
        current = path.join(LABEL_PATH, path_info["path"], file_name)
        with open(current) as labeled:
            label_info = json.load(labeled)
        yolo_lines = transform2yolo(label_id, label_info)

        rand = random.randrange(0, 10)
        div = ""
        if rand < 1:
            div = "test"
            test += 1
        else:
            div = "train"
            train += 1
        cur = os.path.join(YOLO_LABEL_PATH, div)

        name_split = file_name.split(".")
        outlabel_file_name = ".".join(name_split[:-1]) + ".txt"
        with open(path.join(cur, outlabel_file_name), "w") as outfile:
            print("\n".join(yolo_lines), file=outfile)
            print("\n".join(yolo_lines))

        img_name = label_info["Info"]["filename"]

        file_names = img_name.split(".")
        file_names[-1] = file_names[-1].lower()
        img_name = (".").join(file_names)

        current_img = path.join(IMAGE_PATH, path_info["img_path"], img_name)

        img_save_path = path.join(YOLO_IMAGE_PATH, div, img_name)
        # print(f"{current_img}\t{img_save_path}")
        copy2(current_img, img_save_path)
    data_size.append(
        {
            "class": label_id,
            "name": path_info["name"],
            "name_kr": path_info["name_kr"],
            "test": test,
            "train": train,
        }
    )
    total[0] += test
    total[1] += train
info = {"info": {"test": total[0], "train": total[1]}, "data_size": data_size}
with open("dataset_information.json", "w", encoding="UTF-8") as ofile:
    json.dump(info, ofile, ensure_ascii=False, indent=4)

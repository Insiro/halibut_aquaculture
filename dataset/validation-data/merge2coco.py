import os
import json
from os import path

baseDir = "./3_라벨링데이터_coco/11.넙치유생-QL"

categories = [
    {"supercategory": "halibut", "id": 0, "name": "halibut_larva_step1"},
    {"supercategory": "halibut", "id": 1, "name": "halibut_larva_step2"},
    {"supercategory": "halibut", "id": 2, "name": "halibut_larva_step3"},
    {"supercategory": "halibut", "id": 3, "name": "halibut"},
]

merge_divide = int(input("0 : divide\n1 : one file\nselect : ")) == 0


index = 1
annotIndex = 1
images = []
annotations = []

for dir in os.listdir(baseDir):
    dir_path = path.join(baseDir, dir)
    if path.isfile(dir_path):
        continue
    for files in os.listdir(dir_path):
        file = path.join(dir_path, files)
        with open(file) as coco_single:
            single = json.load(coco_single)
            images.append(single["image"])
            annotations.extend(single["annotations"])

    if merge_divide:
        with open(f"{baseDir}/{dir}_coco.json", "w") as coco:
            print(
                json.dumps(
                    {
                        "images": images,
                        "categories": categories,
                        "annotations": annotations,
                    },
                    indent=2,
                ),
                file=coco,
            )
        images = []
        annotations = []

if not merge_divide:
    with open(f"{baseDir}/coco.json", "w") as coco:
        print(
            json.dumps(
                {
                    "images": images,
                    "categories": categories,
                    "annotations": annotations,
                },
                indent=2,
            ),
            file=coco,
        )

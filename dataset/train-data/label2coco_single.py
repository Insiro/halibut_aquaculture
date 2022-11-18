import json
import os
from os import path

PATH_LIST = [
    {"path": "11.넙치유생-QL/LV10", "name_kr": "넙치유생_변태1단계", "name": "halibut_larva_step1"},
    {"path": "11.넙치유생-QL/LV20", "name_kr": "넙치유생_변태2단계", "name": "halibut_larva_step2"},
    {"path": "11.넙치유생-QL/LV30", "name_kr": "넙치유생_변태3단계", "name": "halibut_larva_step3"},
    # {"path": "11.넙치치어-QL", "name_kr": "넙치치어", "name": "halibut"},
]
IMAGE_PATH = "./1_원천데이터"
LABEl_PATH = "./2_라벨링데이터"
COCO_PATH = "./3_라벨링데이터_coco"
label_info = {}
n_annots = 0
n_images = 0

if not path.isdir(COCO_PATH):
    os.mkdir(COCO_PATH)


def transform2coco(
    annot_offset: int, image_id: int, class_id: int, class_name: str, label
):
    annot = label["Annotations"]
    print(json.dumps(annot, indent=4))
    coco_annot = [
        {
            "id": annot_idx + annot_offset,
            "image_id": image_id,
            "bbox": json.loads(annot["rect.points"]),
            "category_id": class_id,
            "segmentation": [],
        }
        for annot_idx, annot in enumerate(annot)
    ]
    # n_annots += len(coco_annot)
    resol = json.loads(label["Info"]["resolution"])
    file_name = label["Info"]["filename"]
    file_names = file_name.split(".")
    file_names[-1] = file_names[-1].lower()
    file_name = (".").join(file_names)
    coco = {
        "image": {
            "id": image_id,
            "file_name": file_name,
            "height": resol[1],
            "width": resol[0],
        },
        "categories": [
            {"suppercategory": "halibut", "id": class_id, "name": class_name}
        ],
        "annotations": coco_annot,
    }
    return coco


for label_id, path_info in enumerate(PATH_LIST):
    new_path = path.join(COCO_PATH, path_info["path"])
    if not path.isdir(new_path):
        cur = COCO_PATH
        for dir in new_path.split("/")[2:]:
            cur = path.join(cur, dir)
            if not path.isdir(cur):
                os.mkdir(cur)
    for file_name in os.listdir(path.join(LABEl_PATH, path_info["path"])):
        current = path.join(LABEl_PATH, path_info["path"], file_name)
        new_file = path.join(new_path, file_name)
        with open(current) as labeled:
            label_info = json.load(labeled)
        coco = transform2coco(
            n_annots, n_images, label_id, path_info["name"], label_info
        )
        n_annots += len(coco["annotations"])
        n_images += 1
        with open(new_file, "w") as outfile:
            json.dump(coco, outfile)

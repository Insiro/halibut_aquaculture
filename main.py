import os
import sys
from pathlib import Path
import numpy as np

import torch
from pandas import DataFrame

sys.path.append(os.path.abspath("./yolov5"))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages

from yolov5.utils.general import (
    LOGGER,
    Profile,
    cv2,
    colorstr,
    check_img_size,
    increment_path,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)


from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, smart_inference_mode

WEIGHT = "configs/best.pt"  # model path or triton URL
SOURCE = "dataset/validation-data/images/val"
DATA = "configs/fish.yaml"  # dataset.yaml path

TARGET_CLASSES = [
    "halibut_larva_step1",
    "halibut_larva_step2",
    "halibut_larva_step3",
    "halibut",
]


@smart_inference_mode()
def detection(
    weights=WEIGHT,  # model path or triton URL
    source=SOURCE,  # file/dir/URL/glob/screen/0(webcam)
    data=DATA,  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_txt=False,  # save results to *.txt
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    project="runs/pred",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
):
    total_instance = []
    source = str(source)

    # Directories
    if not nosave:
        save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok
        )  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(
            parents=True, exist_ok=True
        )  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

    # Run inference
    colorstr,
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, dt = 0, (Profile(), Profile(), Profile())
    for path, im, im0s, _, _ in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, False, max_det=max_det
            )

        # Process predictions
        for det in pred:  # per image
            img_instances = [0, 0, 0, 0, 0]  # ..classes, sum
            seen += 1

            p, im0 = path, im0s.copy()
            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            assert len(det) != 0

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    img_instances[c] += 1
                    if save_txt:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (cls, *xywh)  # label format
                        txt_path = str(save_dir / "labels" / p.stem)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if not nosave:  # Add bbox to image
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            # Save results (image with detections)
            if not nosave:
                save_path = str(save_dir / p.name)  # im.jpg
                cv2.imwrite(save_path, im0)
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/{p.name}")
            img_instances[-1] = sum(img_instances)
            total_instance.append(img_instances)

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}"
        % t
    )
    return np.array(total_instance)


def get_total_number(np_sample, n_instance):
    classrate = np.array(
        [
            np.average(np.divide(np_sample[:, 0], np_sample[:, -1])),
            np.average(np.divide(np_sample[:, 1], np_sample[:, -1])),
            np.average(np.divide(np_sample[:, 2], np_sample[:, -1])),
            np.average(np.divide(np_sample[:, 3], np_sample[:, -1])),
        ]
    )
    classrate = classrate * n_instance
    return classrate


def main():
    ret = detection(nosave=False)
    counts = np.array(get_total_number(ret, 100))
    ret = DataFrame(counts[None], columns=TARGET_CLASSES)
    print(ret)


if __name__ == "__main__":
    main()

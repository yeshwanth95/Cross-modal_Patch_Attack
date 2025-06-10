import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
from tqdm import tqdm


def xywh_to_xyxy(xywh_box):
    """
    Convert a bounding box from (x_center, y_center, width, height) format
    to (x_min, y_min, x_max, y_max) format.

    Parameters:
    - xywh_box: tuple or list of (x_center, y_center, width, height)

    Returns:
    - tuple of (x_min, y_min, x_max, y_max)
    """
    x_center, y_center, width, height = xywh_box
    # x_min = int(round(x_center - width / 2))
    # y_min = int(round(y_center - height / 2))
    # x_max = int(round(x_center + width / 2))
    # y_max = int(round(y_center + height / 2))
    x_min = int(round(x_center))
    y_min = int(round(y_center))
    x_max = int(round(x_center + width))
    y_max = int(round(y_center + height))
    return (x_min, y_min, x_max, y_max)


def main():
    ann = COCO(f"llvip_dataset_50ims/anns.json")
    ims_dir = f"llvip_dataset_50ims/infrared"
    gt_save_dir = f"scratch/llvip_gts_viz"
    im_ids = ann.getImgIds(catIds=ann.getCatIds())
    all_annos = ann.loadAnns(ids=ann.getAnnIds())
    for id in tqdm(im_ids):
        coco_im = ann.loadImgs(ids=[id])
        img = cv2.imread(os.path.join(ims_dir, f"{id}.jpg"))
        annos = [a for a in all_annos if a["image_id"] == id]
        bboxes = [a['bbox'] for a in annos]
        for bbox in bboxes:
            bbox = xywh_to_xyxy(bbox)
            cv2.rectangle(img, bbox[:2], bbox[2:], color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(gt_save_dir, f"{id}.jpg"), img)


if __name__ == "__main__":
    main()


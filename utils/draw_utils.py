import cv2
import numpy as np
import torch


def draw_predictions(image, bbox, prob, out_path):
    if isinstance(image, torch.Tensor):
        _, C, H, W = image.shape
        image = image.cpu().numpy()
        image = image[0].transpose(1, 2, 0)
        if image.max() <= 1.:
            image *= 255  # scale image values to [0, 255]
        image = image.astype(np.uint8).copy()
    else:
        H, W, C = image.shape
        image = image.copy()
    llx, lly, urx, ury = bbox
    cv2.rectangle(image, (llx, ury), (urx, lly), color=(0, 255, 0), thickness=2)
    cv2.putText(image, str(round(prob.item(), 2)), (llx, lly-12), 0, 1e-3 * H, color=(0, 255, 0), thickness=1)
    cv2.imwrite(out_path, image)


def draw_all_predictions(image, bboxes, probs, out_path):
    if isinstance(image, torch.Tensor):
        _, C, H, W = image.shape
        image = image.cpu().numpy()
        image = image[0].transpose(1, 2, 0)
        if image.max() <= 1.:
            image *= 255  # scale image values to [0, 255]
        image = image.astype(np.uint8).copy()
    else:
        H, W, C = image.shape
        image = image.copy()
    for i, bbox in enumerate(bboxes):
        llx, lly, urx, ury = bbox
        if probs[i].item() >= 0.9:
            color = (255, 255, 0)
        else:
            color = (0, 255, 255)
        cv2.rectangle(image, (llx, ury), (urx, lly), color=color, thickness=2)
        cv2.putText(image, str(round(probs[i].item(), 2)), (llx, lly-12), 0, 1e-3 * H, color=color, thickness=1)
    cv2.imwrite(out_path, image)


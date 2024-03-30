import torch
import torchvision.ops as ops


def xywh_to_xyxy(box):
    x, y, w, h = box
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)


def calculate_iou_for_lists(boxes1, boxes2):
    ious = []

    for box1, box2 in zip(boxes1, boxes2):
        box1_xyxy = xywh_to_xyxy(box1)
        box2_xyxy = xywh_to_xyxy(box2)
        iou = ops.box_iou(box1_xyxy.unsqueeze(0),
                          box2_xyxy.unsqueeze(0))  # Compute IoU for a pair of bounding boxes
        ious.append(iou.item())  # Append the IoU value to the list

    return torch.tensor(ious, dtype=torch.float32)  # Convert the list to a PyTorch tensor

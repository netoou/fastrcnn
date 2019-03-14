import copy
import numpy as np

def bbox_resize(bbox, in_imsize, out_imsize):
    # resizing the bounding box adjust to the mother image
    # in_imsize -> out_imsize
    # x coordinates are depending on the width
    # y are depending on the height

    # bbox[0] : x coordintate of top left
    # bbox[1] : y coordinate of top left
    # bbox[2] : width
    # bbox[3] : height

    # Can process batch of bbox
    bboxes = copy.deepcopy(bbox)
    if type(bbox) == list:
        bboxes = np.array(bboxes, dtype=np.float)

    assert len(bboxes.shape) < 3, "Dimension of bbox is larger than 2."

    if len(bboxes.shape) == 1:

        in_height, in_width = in_imsize
        out_height, out_width = out_imsize

        width_param = out_width / in_width
        height_param = out_height / in_height

        bboxes[0] = bbox[0] * width_param
        bboxes[1] = bbox[1] * height_param
        bboxes[2] = bbox[2] * width_param
        bboxes[3] = bbox[3] * height_param
    elif len(bboxes.shape) == 2:

        in_height, in_width = in_imsize
        out_height, out_width = out_imsize

        width_param = out_width / in_width
        height_param = out_height / in_height

        bboxes[:, 0] = bbox[:, 0] * width_param
        bboxes[:, 1] = bbox[:, 1] * height_param
        bboxes[:, 2] = bbox[:, 2] * width_param
        bboxes[:, 3] = bbox[:, 3] * height_param

    return bboxes

def xywhToxyxy(bbox):
    """
    Change coordinates of bounding box
    from x,y,w,h to x1,y1,x2,y2
    :param bbox: input bbox
    :return: target bbox
    """
    bboxes = copy.deepcopy(bbox)
    if type(bbox) == list:
        bboxes = np.array(bboxes, dtype=np.float)

    assert len(bboxes.shape) < 3, "Dimension of bbox should be less than 3."

    if len(bboxes.shape) == 1:
        bboxes[2] = bboxes[0] + bboxes[2]
        bboxes[3] = bboxes[1] + bboxes[3]
    elif len(bboxes.shape) == 2:
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

    return bboxes

def xyxyToxywh(bbox):
    """
    Change coordinates of bounding box
    from x1,y1,x2,y2 to x,y,w,h
    :param bbox: input bbox
    :return: target bbox
    """
    bboxes = copy.deepcopy(bbox)
    if type(bbox) == list:
        bboxes = np.array(bboxes, dtype=np.float)

    assert len(bboxes.shape) < 3, "Dimension of bbox should be less than 3."

    if len(bboxes.shape) == 1:
        bboxes[2] = bboxes[2] - bboxes[0]
        bboxes[3] = bboxes[3] - bboxes[1]
    elif len(bboxes.shape) == 2:
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

    return bboxes


def iou(bbox1, bbox2, eps=1e-8):
    """
    Intersection of union

    :param bbox1: pytorch tensor of bounding box
    :param bbox2: pytorch tensor of bounding box
    :param gamma: zero division protector
    :return: iou score
    """

    # Total areas
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]

    # Intersection coordinates
    ix1 = max(bbox1[0], bbox2[0])
    iy1 = max(bbox1[1], bbox2[1])
    ix2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    iy2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    # Overlapping area
    i_area = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)

    return i_area / (bbox1_area + bbox2_area - i_area + eps)

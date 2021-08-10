"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# DoC: 2021.07.15
-----------------------------------------------------------------------------------
# Description: The script defines the necessary functions for the project
"""


import cv2
import numpy as np
import tensorflow as tf
from config import cfg
from shapely.geometry import Polygon
import tqdm
from matplotlib import pyplot as plt

def load_freeze_layer():
    """ Get output layers of network
            :return
        """
    freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']
    return freeze_layouts


def read_class_names(class_file_name):
    """ Get the names of classes
            :param class_file_name
            :return names (car,van,truck)
    """
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def load_config():
    """ Get the parameters in config file
            :return STRIDES [8, 16, 32]
                    ANCHORS
                    NUM_CLASS
        """
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = get_anchors(cfg.YOLO.ANCHORS)
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS

def get_anchors(anchors_path):
    anchors = np.array(anchors_path)
    return anchors

def image_preprocess(image, target_size, gt_boxes=None):
    """ Process the size of input images to target_size
                :param image  (1001,1001,3)
                :param target_size (608,608,3)
                :param gt_boxes  label of image
                :return image_paded
                        gt_boxes
            """
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, 1] = gt_boxes[:, 1] * scale + dw
        gt_boxes[:, 2] = gt_boxes[:, 2] * scale + dh
        gt_boxes[:, 5] = gt_boxes[:, 5] / 0.15 * scale
        gt_boxes[:, 6] = gt_boxes[:, 6] / 0.15 * scale
        return image_paded, gt_boxes


class Line:
    # ax + by + c = 0
    def __init__(self, p1, p2):
        """
        Args:
            p1: (x, y)
            p2: (x, y)
        """
        self.a = p2[1] - p1[1]  # a = y2 - y1
        self.b = p1[0] - p2[0]  # b = x1 - x2
        self.c = p2[0] * p1[1] - p2[1] * p1[0]  # x2*y1-y2*x1 cross

    def cal_values(self, pts):
        return self.a * pts[:, 0] + self.b * pts[:, 1] + self.c

    def find_intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a * other.b - self.b * other.a
        return tf.constant([(self.b * other.c - self.c * other.b) / w, (self.c * other.a - self.a * other.c) / w])


def get_corners_3d_single(x, y, z, h, w, l, yaw):
    """ Get the corners of one box
                    :param x, y, z, h, w, l, yaw
                    :return box_conners (8,3)
    """
    box_conners = np.zeros((8, 3))
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # front left
    box_conners[0, 0] = x + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[0, 1] = y - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[0, 2] = z - h / 2

    # rear left
    box_conners[1, 0] = x - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[1, 1] = y + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[1, 2] = z - h / 2

    # rear right
    box_conners[2, 0] = x - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[2, 1] = y + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[2, 2] = z - h / 2

    # front right
    box_conners[3, 0] = x + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[3, 1] = y - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[3, 2] = z - h / 2

    box_conners[4, 0] = x + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[4, 1] = y - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[4, 2] = z + h / 2

    # rear left
    box_conners[5, 0] = x - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[5, 1] = y + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[5, 2] = z + h / 2

    # rear right
    box_conners[6, 0] = x - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[6, 1] = y + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[6, 2] = z + h / 2

    # front right
    box_conners[7, 0] = x + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[7, 1] = y - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[7, 2] = z + h / 2
    return box_conners


def get_corners_3d(x, y, z, h, w, l, yaw):
    """ Get the corners of all boxes in a image
                        :param x, y, z, h, w, l, yaw (num_boxes,1)
                        :return  box_conners (num_boxes, 8, 3)
        """
    box_conners = np.zeros((x.shape[0], 8, 3))  # x.size(0): num_boxes
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # front left
    box_conners[:, 0, 0] = x + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 0, 1] = y - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 0, 2] = z - h / 2

    # rear left
    box_conners[:, 1, 0] = x - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 1, 1] = y + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 1, 2] = z - h / 2

    # rear right
    box_conners[:, 2, 0] = x - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 2, 1] = y + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 2, 2] = z - h / 2

    # front right
    box_conners[:, 3, 0] = x + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 3, 1] = y - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 3, 2] = z - h / 2

    box_conners[:, 4, 0] = x + w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 4, 1] = y - w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 4, 2] = z + h / 2

    # rear left
    box_conners[:, 5, 0] = x - w / 2 * sin_yaw - l / 2 * cos_yaw
    box_conners[:, 5, 1] = y + w / 2 * cos_yaw - l / 2 * sin_yaw
    box_conners[:, 5, 2] = z + h / 2

    # rear right
    box_conners[:, 6, 0] = x - w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 6, 1] = y + w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 6, 2] = z + h / 2

    # front right
    box_conners[:, 7, 0] = x + w / 2 * sin_yaw + l / 2 * cos_yaw
    box_conners[:, 7, 1] = y - w / 2 * cos_yaw + l / 2 * sin_yaw
    box_conners[:, 7, 2] = z + h / 2

    return box_conners


def intersection_area(rect1, rect2):
    """Calculate the inter

        :param rect1: vertices of the rectangles (4, 2)
        :param rect2: vertices of the rectangles (4, 2)
        :returns
    """
    # Use the vertices of the first rectangle as, starting vertices of the intersection polygon.
    intersection = rect1
    # Loop over the edges of the second rectangle
    roll_rect2 = tf.roll(rect2, -1, axis=0)

    for p, q in zip(rect2, roll_rect2):
        if len(intersection) <= 2:
            break  # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".
        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = line.cal_values(intersection)
        roll_intersection = tf.roll(intersection, -1, axis=0)
        roll_line_values = tf.roll(line_values, -1, axis=0)
        for s, t, s_value, t_value in zip(intersection, roll_intersection, line_values, roll_line_values):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.find_intersection(Line(s, t))
                new_intersection.append(intersection_point)

        if len(new_intersection) > 0:
            intersection = tf.stack(new_intersection)
        else:
            break

    # Calculate area
    if len(intersection) <= 2:
        return 0.

    return PolyArea2D(intersection)


def PolyArea2D(pts):
    roll_pts = tf.roll(pts, -1, dims=0)
    area = (pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]).sum().abs() * 0.5
    return area


def cvt_box_2_polygon(box):
    """
    :param array: an array of shape [num_conners, 2]
    :return a shapely.geometry.Polygon object
    """

    return Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0)  # 根据有序的点创建多边形，返回一个多边形类


def get_polygons_areas_fix_xyz(boxes, fix_xyz=100.):
    """

        :param box: (num_boxes, 4) -->h, w, l, yaw

        :return boxes_polygons (num_boxes,4,(x,y))
        :return boxes_volumes
        :return low_h, high_h
    """

    n_boxes = boxes.shape[0]
    x = np.full((n_boxes,), fix_xyz)
    y = np.full((n_boxes,), fix_xyz)
    z = np.full((n_boxes,), fix_xyz)
    h = boxes[:, 0]
    w = boxes[:, 1]
    l = boxes[:, 2]
    yaw = boxes[:, 3]

    boxes_conners_3d = get_corners_3d(x, y, z, h, w, l, yaw)
    boxes_polygons = [cvt_box_2_polygon(box_) for box_ in
                      boxes_conners_3d[:, :4, :2]]  # Take (x,y) of the 4 first conners
    boxes_volumes = h * w * l
    low_h = boxes_conners_3d[:, 0, 2]
    high_h = boxes_conners_3d[:, -1, 2]

    return boxes_polygons, boxes_volumes, low_h, high_h


def iou_rotated_box_target_vs_anchor_single(a_polygon, a_volume, a_low_h, a_high_h, tg_polygon, tg_volume, tg_low_h,
                                            tg_high_h):
    """Caculate the iou between a box of label and a box of anchors
        :param  a_polygon,tg_polygon (4,2)
        :param a_volume, a_low_h, a_high_h, tg_volume, tg_low_h, tg_high_h are constant
        :return
        """
    inter_area = a_polygon.intersection(tg_polygon).area
    low_inter_h = max(a_low_h, tg_low_h)
    high_inter_h = min(a_high_h, tg_high_h)
    inter_volume = (high_inter_h - low_inter_h) * inter_area
    iou = inter_volume / (a_volume + tg_volume - inter_volume + 1e-16)
    return iou


def iou_rotated_boxes_targets_vs_anchors(a_polygons, a_volumes, a_low_hs, a_high_hs, tg_polygons, tg_volumes, tg_low_hs,
                                         tg_high_hs):
    """Caculate the iou between each box of label and each box of anchors
            :param  a_polygon (num_anchors,4,2)
            :param tg_polygon (num_boxes,4,2)
            :param a_volume, a_low_h, a_high_h (num_anchors,1)
            :param tg_volume, tg_low_h, tg_high_h (num_boxes,1)
            :return ious (num_anchors,num_boxes)
            """
    num_anchors = len(a_volumes)
    num_targets_boxes = len(tg_volumes)

    ious = np.zeros((num_anchors, num_targets_boxes))

    for a_idx in range(num_anchors):
        for tg_idx in range(num_targets_boxes):
            iou = iou_rotated_box_target_vs_anchor_single(a_polygons[a_idx], a_volumes[a_idx], a_low_hs[a_idx],
                                                          a_high_hs[a_idx], tg_polygons[tg_idx], tg_volumes[tg_idx],
                                                          tg_low_hs[tg_idx], tg_high_hs[tg_idx])
            ious[a_idx, tg_idx] = iou

    return ious

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)

def iou_rotated_single_vs_multi_boxes(single_box, multi_boxes):
    """
    :param pred_box: Numpy array
    :param target_boxes: Numpy array
    :return:
    """

    s_x, s_y, s_z, s_h, s_w, s_l, s_yaw = single_box
    s_volume = s_h * s_w * s_l
    s_conners = get_corners_3d_single(s_x, s_y, s_z, s_h, s_w, s_l, s_yaw)

    s_polygon = cvt_box_2_polygon(s_conners[:4, :2])
    s_low_h = s_conners[0, 2]
    s_high_h = s_conners[-1, 2]

    m_x, m_y, m_z, m_h, m_w, m_l, m_yaw = multi_boxes.transpose(1, 0)
    targets_volumes = m_h * m_w * m_l

    m_boxes_conners = get_corners_3d(m_x, m_y, m_z, m_h, m_w, m_l, m_yaw)
    m_boxes_polygons = [cvt_box_2_polygon(box_[:4, :2]) for box_ in m_boxes_conners]
    m_boxes_low_hs = m_boxes_conners[:, 0, 2]
    m_boxes_high_hs = m_boxes_conners[:, -1, 2]

    ious = []
    for m_idx in range(multi_boxes.shape[0]):
        inter_area = s_polygon.intersection(m_boxes_polygons[m_idx]).area

        low_inter_h = max(s_low_h, m_boxes_low_hs[m_idx])
        high_inter_h = min(s_high_h, m_boxes_high_hs[m_idx])
        inter_h = max(0., high_inter_h - low_inter_h)
        inter_volume = inter_area * inter_h
        iou_ = inter_volume / (s_volume + targets_volumes[m_idx] - inter_volume + 1e-16)
        ious.append(iou_)

    return np.array(ious)


def get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold):
    """  Caculate TP of prediction
        :param outputs (num_boxes,(x,y,z,h,w,l,yaw,conf_score,class))
        :param targets (num_boxes,(label,x,y,z,h,w,l,class))
        :param iou_threshold
        :return:
        """
    pred_boxes = outputs[:, :7]
    pred_scores = outputs[:, 7]
    pred_labels = outputs[:, -1]
    true_positives = np.zeros(pred_boxes.shape[0])
    if len(targets) > 0:
        target_labels = targets[:, 0]
        detected_boxes = []
        target_boxes = targets[:, 1:]
        for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            if len(detected_boxes) == len(targets):
                break
            if pred_label not in [0,1,2]:
                continue

            iou=iou_rotated_single_vs_multi_boxes(pred_box, target_boxes)
            i_m=iou.argmax()
            iou_max=iou[i_m]
            if (iou_max >= iou_threshold) and (i_m not in detected_boxes):
                true_positives[pred_i] = 1
                detected_boxes += [i_m]
    return true_positives, pred_scores, pred_labels,target_labels

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        :param true_positives TP

        :returns The average precision
        """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc /(tpc + fpc)
            p.append(precision_curve[-1])
            ap.append(compute_ap(recall_curve, precision_curve,c))

            # Compute F1 score (harmonic mean of precision and recall)
    plt.title("P-R curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("r_p")
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision,c):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    if  c==0:
        plt.plot(mrec, mpre,label="Car")
    elif c==1:
        plt.plot(mrec, mpre, label="Truck")
    elif c==2:
        plt.plot(mrec, mpre, label="Van")

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap




"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# DoC: 2021.07.15
-----------------------------------------------------------------------------------
# Description: This script defines the Yolov4 architecture, built the Yolo-layer, in this project we predict 8 parameters
x, y, z, w, h, l, yaw, confidence, so the numeber of output channel is 3 * (8 + NUM_CLASS)
"""

import numpy as np
import tensorflow as tf
import utils as utils
import common as common
import backbone as backbone
from config import cfg





def YOLOv4(input_layer, NUM_CLASS):
    """ build yolov4 network
                :param input_layer: [batsize,608,608,3]
                :param NUM_CLASS: 3
                :return [conv_sbbox, conv_mbbox, conv_lbbox]
                :conv_sbbox [batchsize,output_size,output_size,numbers_anchors,(Num_class+x,y,z,w,h,w,l,yaw,conf))
            """
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 8)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 8)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 8)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode_train(conv_output, output_size, NUM_CLASS):
    """ normalize output data
            :param conv_outpu: [num_samples or batch, grid_size, grid_size, num_anchors, 8+num_classes]
            :param NUM_CLASS: 3
            :return pred:[num_samples or batch, grid_size, grid_size, num_anchors, 8+num_classes]
        """
    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], output_size, output_size, 3, 8 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dz,conv_raw_dhdwdl,conv_raw_yaw,conv_raw_conf, conv_raw_prob = \
        tf.split(conv_output, (2,1, 3, 1,1, NUM_CLASS), axis=-1)

    pred_xy = tf.sigmoid(conv_raw_dxdy)
    pred_z=conv_raw_dz
    pred_xyz=tf.concat([pred_xy,pred_z],axis=-1)
    pred_hwl = conv_raw_dhdwdl
    pred_yaw = conv_raw_yaw
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_cls = tf.sigmoid(conv_raw_prob)
    pred = tf.concat([pred_xyz, pred_hwl, pred_yaw, pred_conf, pred_cls], axis=-1)


    return pred


def build_targets(pred, i, target, num_classes):
    """ Built yolo targets to compute loss
        :param pred: [num_samples or batch, grid_size, grid_size, num_anchors, 8+num_classes]
        :param target: [num_boxes, 9]
        :param anchors: [num_anchors, 4]
        :return obj_mask, noobj_mask, tx, ty, tz, th, tw, tl, tyaw,tconf:[batchsize,output_size,output_size,num_anchors]
        :return tcls: [batchsize,output_size,output_size,num_anchors,num_classes]
    """

    anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
    anchors = anchors.reshape((9,4))
    anchors_i = anchors[3*i: 3*i+3]
    sa_polygons, sa_volumes, sa_low_hs, sa_high_hs = utils.get_polygons_areas_fix_xyz(anchors_i, fix_xyz=100.)

    pred_boxes, _, pred_cls = tf.split(pred, (7, 1, num_classes), axis=-1)
    nB = cfg.TRAIN.BATCH_SIZE
    strides = [8, 16, 32]
    nG = int(608 / strides[i])
    nA = 3
    nC = 3

    n_target_boxes = target.shape[0]
    obj_mask = np.full((nB, nG, nG, nA), 0, dtype=np.uint8)
    noobj_mask = np.full((nB, nG, nG, nA), 1, dtype=np.uint8)
    tx = np.full((nB, nG, nG, nA), 0, dtype=np.float)
    ty = np.full((nB, nG, nG, nA), 0, dtype=np.float)
    tz = np.full((nB, nG, nG, nA), 0, dtype=np.float)
    th = np.full((nB, nG, nG, nA), 0, dtype=np.float)
    tw = np.full((nB, nG, nG, nA), 0, dtype=np.float)
    tl = np.full((nB, nG, nG, nA), 0, dtype=np.float)
    tyaw = np.full((nB, nG, nG, nA), 0, dtype=np.float)
    tcls = np.full((nB, nG, nG, nA, nC), 0, dtype=np.float)

    if n_target_boxes > 0:
        b = target[:, 0].astype(np.uint8)
        target_labels = target[:, 1].astype(np.uint8)
        target_boxes = target[:, 2:9]
        tg_polygons, tg_volumes, tg_low_hs, tg_high_hs = utils.get_polygons_areas_fix_xyz(target_boxes[:, 3:7], fix_xyz=100.)

        ious_a_tg = utils.iou_rotated_boxes_targets_vs_anchors(sa_polygons, sa_volumes, sa_low_hs,
                                                         sa_high_hs, tg_polygons, tg_volumes, tg_low_hs,
                                                         tg_high_hs)

        best_n = np.argmax(ious_a_tg, axis=0).astype(np.uint8)

        hwl = target_boxes[:, 3:6]
        h, w, l = np.transpose(hwl)
        yaw = target_boxes[:, 6:7]

        xy = target_boxes[:, :2]  # (num_boxes, 2)
        z = target_boxes[:, 2:3]
        xy_scaled = (xy / strides[i]).astype(np.float)
        x_scaled = xy_scaled[:, 0]
        y_scaled = xy_scaled[:, 1]

        gij = np.floor(xy / strides[i]).astype(np.uint8)
        gi, gj = np.transpose(gij)
        obj_mask[b, gj, gi, best_n] = 1
        noobj_mask[b, gj, gi, best_n] = 0

        ignore_thresh = 0.3
        for i, anchor_ious in enumerate(np.transpose(ious_a_tg)):
            noobj_mask[b[i], gj[i], gi[i], anchor_ious > ignore_thresh] = 0

        tx[b, gj, gi, best_n] = x_scaled - gi
        ty[b, gj, gi, best_n] = y_scaled - gj
        tz[b, gj, gi, best_n] = z.flatten()
        th[b, gj, gi, best_n] = np.log(h / anchors[best_n][:, 0] + 1e-16)
        tw[b, gj, gi, best_n] = np.log(w / anchors[best_n][:, 1] + 1e-16)
        tl[b, gj, gi, best_n] = np.log(l / anchors[best_n][:, 2] + 1e-16)
        tyaw[b, gj, gi, best_n] = yaw.flatten()
        tcls[b, gj, gi, best_n, target_labels] = 1



    obj_mask = obj_mask.astype(np.bool)
    tconf = obj_mask.astype(np.float)
    noobj_mask = noobj_mask.astype(np.bool)

    return obj_mask, noobj_mask, tx, ty, tz, th, tw, tl, tyaw, tcls, tconf



def evaluate_map(image, bboxes,model):
    """ Computer the
            :param pred: [num_samples or batch, grid_size, grid_size, num_anchors, 8+num_classes]
            :param target: [num_boxes, 9]
            :param anchors: [num_anchors, 4]
            :return obj_mask, noobj_mask, tx, ty, tz, th, tw, tl, tyaw,tconf:[batchsize,output_size,output_size,num_anchors]
            :return tcls: [batchsize,output_size,output_size,num_anchors,num_classes]
        """
    img = np.array([image])
    pred_result = model(img, training=False)
    strides, anchors, NUM_CLASS = utils.load_config()
    anchors = np.array(anchors).reshape(3, 3, 4)
    xy = np.ones(2, )
    z = np.ones(1, )
    hwl = np.ones(3, )
    yaw = np.ones(1, )
    conf = np.ones(1, )
    cls = np.ones(3, )
    for i in range(3):
        pred = pred_result[2 * i + 1]

        xy_grid = tf.meshgrid(tf.range(608 / strides[i]), tf.range(608 / strides[i]))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # tf.stack(xy_grid, axis=-1): (gx, gy, 2)

        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(pred)[0], 1, 1, 3, 1])

        xy_grid = tf.cast(xy_grid, tf.float32)
        conv_raw_dxdy, conv_raw_dz, conv_raw_dhdwdl, conv_raw_yaw, conv_raw_conf, conv_raw_prob = \
            tf.split(pred, (2, 1, 3, 1, 1, NUM_CLASS), axis=-1)
        predd_xy = conv_raw_dxdy + xy_grid
        predd_xy = tf.reshape(predd_xy, (-1, 2))
        predd_xy = np.array(predd_xy) * strides[i]
        xy = np.vstack((xy, predd_xy))
        predd_z = tf.reshape(conv_raw_dz, -1)
        z = np.hstack((z, predd_z))
        anchors_scale = anchors[i, :, :3]
        predd_hwl = np.exp(conv_raw_dhdwdl) * anchors_scale
        predd_hwl = predd_hwl.reshape(predd_xy.shape[0], 3)
        hwl = np.vstack((hwl, predd_hwl))
        predd_yaw = conv_raw_yaw
        predd_yaw = tf.reshape(predd_yaw, (-1,))
        predd_yaw = np.array(predd_yaw)
        yaw = np.hstack((yaw, predd_yaw))
        predd_conf = conv_raw_conf
        predd_conf = tf.reshape(predd_conf, (-1,))
        predd_conf = np.array(predd_conf)
        conf = np.hstack((conf, predd_conf))
        predd_cls = conv_raw_prob
        predd_cls = tf.reshape(predd_cls, (-1, 3))
        predd_cls = np.array(predd_cls)
        cls = np.vstack((cls, predd_cls))
    xy = xy[1:]

    z = z[1:]
    hwl = hwl[1:]
    yaw = yaw[1:]
    conf = conf[1:]
    cls = cls[1:]
    f_i = np.where(conf >= 0.99999)
    conf_f = conf[f_i].flatten()
    cls_f = cls[f_i, :].reshape(conf_f.shape[0], 3)
    score = conf_f * cls_f.max(axis=1)
    ii = np.argsort(-score)
    score = score[ii]
    cls_f = cls_f[ii]
    conf_f = conf_f[ii]
    x_f = (xy[f_i, 0].flatten()) * 1001 / 608
    x_f = x_f[ii]
    y_f = (xy[f_i, 1].flatten()) * 1001 / 608
    y_f = y_f[ii]
    z_f = z[f_i]
    z_f = z_f[ii]
    h_f = hwl[f_i, 0].flatten()
    h_f = h_f[ii]
    w_f = hwl[f_i, 1].flatten() * 1001 / 608
    w_f = w_f[ii]
    l_f = hwl[f_i, 2].flatten() * 1001 / 608
    l_f = l_f[ii]
    yaw_f = yaw[f_i]
    yaw_f = yaw_f[ii]
    box_conners = utils.get_corners_3d(x_f, y_f, z_f, h_f, w_f, l_f, yaw_f)

    boxes_polygons = [utils.cvt_box_2_polygon(box_) for box_ in
                      box_conners[:, :4, :2]]  # Take (x,y) of the 4 first conners
    boxes_volumes = h_f * w_f * l_f
    low_h = box_conners[:, 0, 2]
    high_h = box_conners[:, -1, 2]

    def iou_rotated_box_target_vs_anchor_single(a_polygon, a_volume, a_low_h, a_high_h, tg_polygon, tg_volume, tg_low_h,
                                                tg_high_h):
        inter_area = a_polygon.intersection(tg_polygon).area
        low_inter_h = max(a_low_h, tg_low_h)
        high_inter_h = min(a_high_h, tg_high_h)
        inter_volume = (high_inter_h - low_inter_h) * inter_area
        iou = inter_volume / (a_volume + tg_volume - inter_volume + 1e-16)
        return iou

    for i in range(box_conners.shape[0]):
        for j in range(box_conners.shape[0]):
            if i == j:
                continue
            elif np.isnan(box_conners[i]).any():
                continue
            elif np.isnan(box_conners[j]).any():
                continue
            ious = iou_rotated_box_target_vs_anchor_single(boxes_polygons[i], boxes_volumes[i], low_h[i], high_h[i],
                                                           boxes_polygons[j], boxes_volumes[j], low_h[j], high_h[j])
            if ious >= 0.001:
                box_conners[j] = None
    ia = np.argwhere(np.isnan(box_conners[:, 1, 1]) == False)
    ia = ia.flatten()

    score = score[ia]
    cls_f = cls_f[ia]
    zz = z_f[ia]
    classes = cls_f.argmax(axis=1)
    classes=classes.flatten()
    yaw_f = yaw_f[ia]
    output=np.vstack((x_f[ia],y_f[ia],zz,h_f[ia],w_f[ia],l_f[ia],yaw_f,score,classes))

    output=output.T

    true_positives, pred_scores, pred_labels,target_labels=utils.get_batch_statistics_rotated_bbox(output, bboxes, 0.4)
    return true_positives, pred_scores, pred_labels,target_labels




















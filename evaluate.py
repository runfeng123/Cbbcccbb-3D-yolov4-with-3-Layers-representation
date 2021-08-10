"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# DoC: 2021.07.15
-----------------------------------------------------------------------------------
# Description: This script defines the main evaluation function based on validation dataset
"""


from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
from yolov4 import YOLOv4, decode_train, build_targets,evaluate_map
from data_validation import Dataset
from config import cfg
import numpy as np
import utils
import os
flags.DEFINE_string('weights',"./checkpoints/yolov4", 'pretrained weights')
def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
    tf.config.set_visible_devices(physical_devices[1:2], "GPU")
    os.environ["CUDA_VISBLE_DEVICES"] = ""
    validaset = Dataset()
    input_layer = tf.keras.layers.Input([cfg.VALIDATION.INPUT_SIZE, cfg.VALIDATION.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS = utils.load_config()

    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS)
        elif i == 1:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS)
        else:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS)
        bbox_tensors.append(fm)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)

    if FLAGS.weights == None:
        print("evaluate from scratch")
    else:
        model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)
    model.summary()
    true_positivess=np.zeros(1,)
    pred_scoress=np.zeros(1,)
    pred_labelss=np.zeros(1,)
    target_labelss=np.zeros(1,)
    n=len(validaset)
    for i in range(10):
        image, bboxes = validaset.__next__()

        bboxes[:,[1,2,5,6]]=bboxes[:,[1,2,5,6]]*1001/608

        true_positives, pred_scores, pred_labels,target_labels=evaluate_map(image, bboxes,model)
        true_positivess=np.hstack((true_positivess,true_positives))
        pred_scoress=np.hstack((pred_scoress,pred_scores))
        pred_labelss=np.hstack((pred_labelss,pred_labels))
        target_labelss=np.hstack((target_labelss,target_labels))

    precision, recall, AP, f1, ap_class =utils.ap_per_class(true_positivess[1:], pred_scoress[1:], pred_labelss[1:], target_labelss[1:])
    print(AP.shape)
    print("\nmAP: {}\n".format(AP.mean()))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
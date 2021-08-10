"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# DoC: 2021.07.15
-----------------------------------------------------------------------------------
# Description: Generate input data for validating model
"""
import tensorflow as tf
from config import cfg
import numpy as np
import utils
import cv2
import json

class Dataset(object):
    def __init__(self):
        self.strides, self.anchors, NUM_CLASS = utils.load_config()
        self.batch_size = cfg.VALIDATION.BATCH_SIZE

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        with open(cfg.VALIDATION.txt_PATH_PATH , "r") as f:
            self.annotations = json.load(f)

        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

        with tf.device("/cpu:0"):
            train_reader = tf.data.TFRecordDataset(cfg.VALIDATION.ANNOT_PATH)
            self.cc = train_reader.map(self._parse_function)
            self.kk=iter(self.cc)

    def _parse_function(self,exam_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            # 'image' : tf.io.FixedLenSequenceFeature([], tf.string,allow_missing=True, default_value="b"),
            'label': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=None),

        }
        temp = tf.io.parse_example(exam_proto, feature_description)
        img = tf.image.decode_png(temp['image'], channels=3)
        # img = tf.reshape(img, [1001,1001, 3])
        labels = tf.reshape(temp['label'], [-1, 8])

        return img, labels
    def __iter__(self):
        return self
    def __next__(self):
        with tf.device("/cpu:0"):
            if self.batch_count < self.num_batchs:
                    image, bboxes = self.kk.__next__()
                    image=image.numpy()
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    bboxes=bboxes.numpy()
                    image, bboxes = utils.image_preprocess(
                        np.copy(image),
                        [608, 608],
                        np.copy(bboxes),
                    )
                    return (
                        image, bboxes
                )
            else:
                self.batch_count = 0
                self.cc=self.cc.shuffle(900)
                self.kk=iter(self.cc)
                raise StopIteration

    def __len__(self):
        return self.num_batchs


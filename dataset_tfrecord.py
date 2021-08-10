"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# DoC: 2021.07.15
-----------------------------------------------------------------------------------
# Description: Generate input data for training model
"""

import cv2
import numpy as np
import tensorflow as tf
import utils
from config import cfg

import json


class Dataset(object):
    """implement Dataset here
       :return batch_image (batchsize,input_size,input_size,3)
               target (num_boxes,(batch_num,class,x,y,z,h,w,l,yaw))
    """


    def __init__(self):
        self.strides, self.anchors, NUM_CLASS = utils.load_config()
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        #with open("train_label.txt", "r") as f:
            #self.annotations = json.load(f)
        with open(cfg.TRAIN.txt_PATH, "r") as f:
            self.annotations = json.load(f)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

        with tf.device("/cpu:0"):
            train_reader = tf.data.TFRecordDataset(cfg.TRAIN.ANNOT_PATH)
            self.cc = train_reader.map(self._parse_function)
            self.kk=iter(self.cc)

    def _parse_function(self,exam_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=None),

        }
        temp = tf.io.parse_example(exam_proto, feature_description)
        img = tf.image.decode_png(temp['image'], channels=3)
        labels = tf.reshape(temp['label'], [-1, 8])

        return img, labels
    def __iter__(self):
        return self
    def __next__(self):
        with tf.device("/cpu:0"):
            self.train_input_size = cfg.TRAIN.INPUT_SIZE
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3,
                ),
                dtype=np.float32,
            )


            num = 0
            c = np.zeros((1, 9))
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    try:
                        image, bboxes = self.kk.__next__()
                    except:
                        self.kk=iter(self.cc)
                        image, bboxes = self.kk.__next__()

                    else:
                        image=image
                        bboxes=bboxes
                    image=image.numpy()
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    bboxes=bboxes.numpy()
                    image, bboxes = utils.image_preprocess(
                        np.copy(image),
                        [608, 608],
                        np.copy(bboxes),
                    )

                    num_bboxes = len(bboxes)
                    a = np.full((num_bboxes,1),num)
                    b = np.hstack((a, bboxes))
                    c = np.vstack((c, b))
                    batch_image[num, :, :, :] = image
                    num += 1
                self.batch_count += 1
                target = c[1:]
                return (
                    batch_image,
                    target
                )
            else:
                self.batch_count = 0
                self.cc=self.cc.shuffle(900)
                self.kk=iter(self.cc)
                raise StopIteration

    def __len__(self):
        return self.num_batchs



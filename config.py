"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# DoC: 2021.07.15
-----------------------------------------------------------------------------------
# Description: The configurations of the project will be defined here
"""
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C


__C.YOLO                      = edict()
__C.YOLO.repre_path           ="./data/dataset/zparameters.txt"
__C.YOLO.CLASSES              = "./data/classes/detection.names.txt"
__C.YOLO.ANCHORS              = [1.57, 6, 13, 0, 1.42, 6, 14,0, 1.5, 7, 15,0, 1.65, 7, 16, 0,   1.43, 7, 17, 0,1.5, 7, 18, 0, 1.9, 8, 18, 0,2.52,8,24,0,3.52,11,44,0]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3

__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/train_data"
__C.TRAIN.txt_PATH            ="./data/dataset/train_label.txt"
__C.TRAIN.BATCH_SIZE          = 2
__C.TRAIN.INPUT_SIZE          = 608
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30



# TEST options
__C.VALIDATION                      = edict()

__C.VALIDATION.ANNOT_PATH           = "./data/dataset/validation_data"
__C.VALIDATION.BATCH_SIZE           = 1
__C.VALIDATION.INPUT_SIZE           = 608
__C.VALIDATION.DATA_AUG             = False
__C.VALIDATION.txt_PATH_PATH = "./data/dataset/validation_label.txt"




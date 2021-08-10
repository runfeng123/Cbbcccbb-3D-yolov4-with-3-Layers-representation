"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# DoC: 2021.07.15
-----------------------------------------------------------------------------------
# Description: This script defines the main function of training
"""

from absl import app, flags
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from yolov4 import YOLOv4, decode_train, build_targets
from dataset_tfrecord import Dataset
from config import cfg
import numpy as np
import utils
from utils import freeze_all, unfreeze_all

flags.DEFINE_string('weights',None, 'pretrained weights')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
    tf.config.set_visible_devices(physical_devices[1:2],"GPU")
    os.environ["CUDA_VISBLE_DEVICES"]=""
    trainset = Dataset()
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = 20
    second_stage_epochs = 30
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = tf.cast(steps_per_epoch * cfg.TRAIN.WARMUP_EPOCHS, dtype = tf.int64)
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS = utils.load_config()

    freeze_layers = utils.load_freeze_layer()

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
    #input_layer:[batchsize, input_size,input_size,3]
    model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)

    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    def train_step(image_data, target, epoch):
        with tf.GradientTape() as tape:

            pred_result = model(image_data, training=True)
            total_loss = tf.constant(0, dtype=tf.float64)
            xyzwhl_loss = tf.constant(0, dtype=tf.float64)
            yaw_loss=tf.constant(0, dtype=tf.float64)
            conf_loss=tf.constant(0, dtype=tf.float64)
            prob_loss=tf.constant(0, dtype=tf.float64)
            #computer loss of each batchsize
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]

                obj_mask, noobj_mask, tx, ty, tz, th, tw, tl, tyaw, tcls, tconf \
                    = build_targets(pred, i, target, 3)
                pred_x = pred[...,0]
                pred_y = pred[...,1]
                pred_z = pred[...,2]
                pred_h = pred[...,3]
                pred_w = pred[...,4]
                pred_l = pred[...,5]
                pred_yaw = pred[...,6]

                pred_conf = pred[...,7]
                pred_cls = pred[...,8:]


                loss_x = tf.losses.mean_squared_error(pred_x[obj_mask], tx[obj_mask])
                loss_y = tf.losses.mean_squared_error(pred_y[obj_mask], ty[obj_mask])
                loss_z = tf.losses.mean_squared_error(pred_z[obj_mask], tz[obj_mask])
                loss_h = tf.losses.mean_squared_error(pred_h[obj_mask], th[obj_mask])
                loss_w = tf.losses.mean_squared_error(pred_w[obj_mask], tw[obj_mask])
                loss_l = tf.losses.mean_squared_error(pred_l[obj_mask], tl[obj_mask])
                loss_yaw=tf.reduce_mean(tf.sin(pred_yaw[obj_mask]-tyaw[obj_mask])**2)
                loss_yaw=tf.cast(loss_yaw,tf.float64)
                loss_conf_noobj = tf.keras.losses.binary_crossentropy(pred_conf[noobj_mask], tconf[noobj_mask])
                loss_conf_obj = tf.keras.losses.binary_crossentropy(pred_conf[obj_mask], tconf[obj_mask])
                loss_cls=tf.keras.losses.binary_crossentropy(pred_cls[obj_mask], tcls[obj_mask])
                obj_scale = 1
                noobj_scale = 100
                xyzwhl_loss+=loss_x + loss_y + loss_z + loss_h + loss_w + loss_l
                yaw_loss+=loss_yaw
                loss_box = loss_x + loss_y + loss_z + loss_h + loss_w + loss_l + loss_yaw

                loss = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj + loss_cls + loss_box
                conf_loss+=obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj
                prob_loss+=tf.reduce_mean(loss_cls)

                total_loss += tf.reduce_mean(loss)


            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            tf.print("=> STEP %4d/%4d   lr: %.8f  " " total_loss: %4.2f " " xyzwhl_loss: %4.2f " " yaw_loss: %4.2f " " conf_loss: %4.2f " "  prob_loss: %4.2f " " epoch: %4d "% (global_steps, total_steps, optimizer.lr.numpy(), total_loss, xyzwhl_loss,yaw_loss,conf_loss,prob_loss,epoch))

            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)

            writer.flush()



    for epoch in range(50):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)

        for image_data, target in trainset:
            train_step(image_data, target, epoch)

        model.save_weights("./checkpoints_ls{}/yolov4".format(epoch))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
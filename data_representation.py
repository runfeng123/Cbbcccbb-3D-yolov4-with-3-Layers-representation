"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# DoC: 2021.07.15
-----------------------------------------------------------------------------------
# Description: representation The rotated representation(22.5°*16) of the Point cloud data
               velodyne file need to be downloaded we didn't upload it
"""


import numpy as np
import cv2
import json
import multiprocessing
from config import cfg


with open(cfg.YOLO.repre_path,"r") as file:
    zf=json.load(file)
zn,z1,z2,zx=zf
def image_process(x):

    file_name = '%06d' % x
    pointcloud = np.fromfile(str("velodyne/{}.bin".format(file_name)), dtype=np.float32, count=-1).reshape([-1, 4])
    side_range = (-75, 75)
    fwd_range = (-75, 75)
    x_points = pointcloud[:, 0]
    y_points = pointcloud[:, 1]
    z_points = pointcloud[:, 2]
    z_points=np.tile(z_points,(16,1)).T
    th=np.linspace(0,30*np.pi/16,16)
    x_points1=np.cos(th)*x_points[:,np.newaxis]-np.sin(th)*y_points[:,np.newaxis]
    y_points1=np.sin(th)*x_points[:,np.newaxis]+np.cos(th)*y_points[:,np.newaxis]
    x_points=x_points1
    y_points=y_points1
    f_filt = np.logical_and(x_points >=fwd_range[0], x_points <=fwd_range[1])
    s_filt = np.logical_and(y_points >= side_range[0], y_points <= side_range[1])
    z_filt=np.logical_and(z_points >= zn, z_points <= zx)
    filter1 = np.logical_and(f_filt, s_filt)
    filter = np.logical_and(filter1, z_filt)
    #indices = np.argwhere(filter).flatten()
    for h in range(16):
        x_points1 = x_points[:,h][filter[:,h]]
        y_points1 = y_points[:,h][filter[:,h]]
        z_points1 = z_points[:,h][filter[:,h]]
        res=0.15
        xmp=np.floor((-y_points1+75)/res).astype(np.int32)
        ymp=np.floor((-x_points1+75)/res).astype(np.int32)
        tt1=np.vstack([xmp,ymp])
        i1=np.zeros((1001,1001))
        i2=np.zeros((1001,1001))
        i3=np.zeros((1001,1001))
        for i,j in enumerate(tt1.T):
            if z_points1[i]>=zn and z_points1[i]<=z1:
                i1[j[1],j[0]]+=1
            elif z_points1[i]>z1 and z_points1[i]<=z2:
                i2[j[1], j[0]] += 1
            elif z_points1[i]>z2 and z_points1[i]<=zx:
                i3[j[1],j[0]]+=1
        i1= np.where(i1>=100,100,i1)/100
        i2= np.where(i2>=100,100,i2)/100
        i3= np.where(i3>=100,100,i3)/100
        RGBimg = cv2.merge([i1,i2,i3])*255

        image_name = '%06d' % x
        thred='%02d' % h
        print(RGBimg.max())
        cv2.imwrite("./{}({}).png".format(image_name,thred),RGBimg)

if __name__=="__main__":

    po=multiprocessing.Pool(3)
    for x in range(1):
       po.apply_async(image_process,(x,))
    po.close()
    po.join()


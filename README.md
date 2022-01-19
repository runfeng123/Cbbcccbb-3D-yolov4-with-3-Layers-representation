- # Yolov4 3D bounding box detection based on Pointcloud 


  ### Dataset

  ##### Kitti dataset Kitti dataset

  the data need to be download includes:

  - Velodyne point clouds (29 GB)
  - Training labels of object data set (5 MB)

  #### representation of data

  Real world area we want to cover: 150m x 150m
  
  Devide the data into 3 layers along the z axis
  
  Each layer represents a channel of RGB image
  
  Partition of grid(x,y): 0.15m 
  
  Grid size: 1001x1001x3
  
  Image size: 1001 x 1001 pixels (RGB image with 3 channels)

  The values of each pixel on images: The number of poins in each block
  

  ### input and prediction

  - input: 608* 608* 3
  - outout: x,y,z,h,w,l,yaw,confidence,class
  - class: Car, Truck, Van

  ### how to start

  ##### train

      bash train.sh 

  ##### evaluate

      bash evaluate.sh

  ##### data representation

      python data_representation.py

  ##### get anchors of data

      python get_anchors.py

  ### Structure

```
${ROOT}
      └── checkpoints
      └── data/
      │   └── classes/
      │   │   ├──detection.names
      │   ├── dataset
      │   │   │   ├── train_data
      │   │   │   └──  train_label.txt
      │   │   │   ├── validation_data
      │   │   │   ├── validation_label.txt
      │   │   │   ├── zparameters.txt
      │   │   │   ├── noRlabel.txt
      ├── backbone.py
      ├── common.py
      ├── config.py
      ├── data_representation.py
      ├── data_validation.py
      ├── dataset_tfrecord.py
      ├── evaluate.py
      ├── get_anchors.py
      ├── train.py
      ├── utils.py
      ├── yolov4.py
      ├── train.sh
      ├── evaluate.sh
  	  ├── demo.png
      ├── README.md
      └── requirements.txt
```

  ##### performance
  Map: 76.8
  
  FPS:35


  ### reference

  [https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch/tree/4e7f65fe25198f2202f8317fa88321729821ceb8](https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch/tree/4e7f65fe25198f2202f8317fa88321729821ceb8)

  [https://github.com/hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)

  


  ### author

  - Qu Runfeng

  - Wu Hao

  - Jiao Weiqin

  ### Guiders

  - Sharma, Kishan




# INPIOD-Pytorch Code backup

## Introduction

Code for the initial experiment in INPIOD dataset (Complemented with Pytorch). 

Only add a instance embedding branch based on C2Net.

## Environment

Please use the anaconda 3.7 and run: 

```
conda create --name c2net --file specf_c2net.txt
```


## Data Preparation

You should set the root path of the dataset to ```data/INPIOD```, or set the dataset path in  ```experiments/configsPIOD_myUnet_kxt.yaml```.

You may download the dataset [original images](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar) and [annotations](https://1drv.ms/u/s!AhUctW17cZ9Gc5O8WITxfvWd2LM?e=qLP4yO). Then you should copy or move all images to`data/INPIOD/JPEGImages` folder. Then unzip ``Annotation_mat.zip`` to `data/INPIOD/Annotation_mat/` folder. You will have the following directory structure:

```
INPIOD
|_ Annotation_mat
|  |_ <id-1>.mat
|  |_ ...
|  |_ <id-n>.mat
|_ JPEGImages 
|  |_ <id-1>.jpg
|  |_ ...
|  |_ <id-n>.jpg
|_ trainval_INPIOD.txt
|_ train_INPIOD.txt
|_ val_INPIOD.txt
```

#### 


## Training & Evaluation

For training, you can run:

```
cd $ROOT/detect_occ/
python train_val_lr.py --config ../experiments/configs/PIOD_myUnet_cy.yaml --gpus 0
```

You can also download the [pretrained model](https://pan.baidu.com/s/1mGchc2lI__HE4KIbvyuGIg) (baiduyun code:pwfi) of INPIOD and put it in  ```experiments/output/```. Then you can evaluate the results by runing: 
```
python train_val_lr.py --config ../experiments/configs/PIOD_myUnet_cy.yaml --evaluate --resume 2021-03-16_14-05-08/checkpoint_19.pth.tar --gpus 0
```

Then the result files in ``.mat`` format are saved in ``experiments/output/2021-03-16_14-05-08/result_vis_KMeans``. Besides, the instance boundary visualizations in color-edge are saved in ``experiments/output/2021-03-16_14-05-08/result_vis_KMeans``.


# DNN-centerNet

![PyTest](https://github.com/yxyfer/DNN-centerNet/actions/workflows/pytest.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![HitCount](https://hits.dwyl.com/yxyfer/DNN-centerNet.svg?style=flat-square)](http://hits.dwyl.com/yxyfer/DNN-centerNet)

![alt text](images/image.png "Mnist Detection example")

## Introduction

This project is a re-implementation of CenterNet in PyTorch for detecting digits in the Mnist Detection dataset. To complete this project, we relied on the paper ["CenterNet: Keypoint Triplets for Object Detection"](https://arxiv.org/pdf/1904.08189.pdf) as well as the following GitHubs: [CenterNet's Repository](https://github.com/Duankaiwen/CenterNet) and [zzzxxxttt's implementation of centerNet](https://github.com/zzzxxxttt/pytorch_simple_CenterNet_47).



To train this model, we first trained a backbone to classify the digits of the MNIST dataset. With the pre-trained backbone, we then trained the CenterNet model for 90 epochs (took ~2h30). The training was done using the MNIST Detection dataset with images of size 300x300 that can contain up to 30 digits per image. The final results of the training are as follows:


|       | mIoU   | AA<sub>5</sub> | AA<sub>50</sub>  | AA<sub>75</sub> | AA<sub>95</sub> | FD<sub>5</sub> | FD<sub>50</sub> | FD<sub>75</sub> | FD<sub>95</sub> |
| ----- | ------ | ------- | ------- | ------- | ------- | ------- | ------ | ------ | ------- |
| Train | 0.9129 | 99.16%  | 98.75%  | 96.79%  | 31.29%  | 0.48%   | 1.71%  | 3.76%  | 68.90%  |
| Test  | 0.8568 | 85.09%  | 82.34%  | 78.56%  | 19.63%  | 4.88%   | 8.17%  | 12.65% | 78.10%  |

- AA = Average Accuracy
- FD = False Detection

## Prerequisites

### Corner Pooling Layers

Before running this project, you must first compile the Corner Pooling Layers. Follow these steps:

Access the cpools\_ folder:

```bash
cd <DNN-centerNet dir>/src/center_net/cpools_/
```

If you want to use a CPU version, run the following command:
```bash
python setup_cpu.py install --user
```

If you want to use a GPU version, run the following command:
```bash
python setup_gpu.py install --user
```

### Dataset

The dataset used for this project is the Mnist Detection dataset. It can be downloaded using this project: [MNIST-ObjectDetection](https://github.com/hukkelas/MNIST-ObjectDetection)

For this project we used the following command to generate the dataset:

```bash
 python generate_data.py --max-digits-per-image 30 --imsize 300
```

## Usage

This project is divided into two parts: the backbone and the CenterNet (which uses the backbone). The backbone is a small CNN trained with Mnist data. It allows CenterNet to have a first representation of what a digit is.

All models are saved in the folder: _models_.

### Backbone

A model of the backbone is already trained and available, however you can retrain it using the following command:
```bash
python train_backbone.py [OPTIONS]
```
The command has several options that can be used to modify the behavior of the training process:

- _--name NAME_: This option sets the name of the model that will be saved at the end of the training. The default name is "backbone_model.pth"
- _--epochs EPOCHS_: This option sets the number of epochs for which the model will be trained. The default value for this option is 20.
- _--batch_size BATCH_SIZE_: This option sets the batch size that will be used during training. The default value for this option is 64.
- _--keep_best_: This option determines whether the best performing model will be saved during the training. By default, this option is set to false, which means that the best model will not be saved.

### CenterNet

Similarly, a trained CenterNet model is also available, however you can retrain it using the following command:
```bash
python train_center_net.py [OPTIONS]
```
The command has several options that can be used to modify the behavior of the training process:

- _--name NAME_: This option sets the name of the model that will be saved at the end of the training. The default name is "center_net_model.pth"
- _--epochs EPOCHS_: This option sets the number of epochs for which the model will be trained. The default value for this option is 90.
- _--dataset DATASET_: This option sets the path to the dataset that will be used for training the model. The default path is "data/mnist_detection".
- _--batch_size BATCH_SIZE_: This option sets the batch size that will be used during training. The default value for this option is 8.
- _--keep_best_: This option determines whether the best performing model will be saved during the training. By default, this option is set to false, which means that the best model will not be saved.
- _--max_objects MAX_OBJECTS_: This option sets the maximum number of objects that can be present in an image. The default value for this option is 30.
- _--max_images_train MAX_IMAGES_TRAIN_: This option sets the maximum number of images that will be used for training. The default value for this option is 700.
- _--max_images_test MAX_IMAGES_TEST_: This option sets the maximum number of images that will be used for testing. The default value for this option is 200.
- _--pretrained_backbone PRETRAINED_BACKBONE_: This option sets the path to the pre-trained backbone model that will be used for training the center-net model. If set to 'none', no pre-trained model will be loaded. The default path is "models/backbone_model.pth".


**remark**:
The _--dataset_ argument must be a folder having the following structure:

```bash
dataset/
|--- train/
|    |--- images/
|         |--- 0.png
|         |--- 1.png
|         |--- 2.png
|         |--- ...
|    |--- labels/
|         |--- 0.txt
|         |--- 1.txt
|         |--- 2.txt
|         |--- ...
|--- test/
|    |--- images/
|         |--- 0.png
|         |--- 1.png
|         |--- ...
|    |--- labels/
|         |--- 0.txt
|         |--- 1.txt
|         |--- ...
```

To test the model on an image, you can use the following command:
```bash
python tester_center_net.py [OPTIONS] image
```
The command has several options that can be used to modify the behavior of the model during testing: 

- _--min_global_score MIN_GLOBAL_SCORE_ : This option sets the minimum score for plotting the bounding boxes. The default value for this option is 0.
- _--min_score MIN_SCORE_ : This option sets the minimum score for a detection to be considered. The default value for this option is 0.05.
- _--not_use_center_ : This option determines whether the center region will be used for filtering the detections. By default, this option is set to False, which means that the center region will be used
- _--center_ : This option is used to display the centers of the bounding boxes in the output image. By default, this option is set to False, which means that the centers will not be displayed.
- _--K K_ : This option sets the number of centers that will be detected by the model. The default value for this option is 70.
- _--num_dets NUM_DETS_ : This option sets the number of detections that will be generated by the model. The default value for this option is 500.
- _--n N_ : This option sets the scale of the central region. The value of N must be an odd number and can be either 3 or 5. The default value for this option is 5.
- _--model MODEL_ : This option sets the path to the trained model that will be used for testing. The default path is models/center_net_model.pth
- _image_: This argument represents the path to the image that will be used.
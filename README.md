# DNN-centerNet

![PyTest](https://github.com/yxyfer/DNN-centerNet/actions/workflows/pytest.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![HitCount](https://hits.dwyl.com/yxyfer/DNN-centerNet.svg?style=flat-square)](http://hits.dwyl.com/yxyfer/DNN-centerNet)

![alt text](images/image.png "Mnist Detection example")

## Introduction

This project is a re-implementation of CenterNet in PyTorch for detecting digits in the Mnist Detection dataset. To complete this project, we relied on the paper ["CenterNet: Keypoint Triplets for Object Detection"](https://arxiv.org/pdf/1904.08189.pdf) as well as the following GitHubs: https://github.com/Duankaiwen/CenterNet and https://github.com/zzzxxxttt/pytorch_simple_CenterNet_47.

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

The dataset used for this project is the Mnist Detection dataset. It can be downloaded using this project: https://github.com/hukkelas/MNIST-ObjectDetection

For this project we used the following command to generate the dataset:

```bash
 python generate_data.py --max-digits-per-image 15 --imsize 300
```

## Usage

This project is divided into two parts: the backbone and the CenterNet (which uses the backbone). The backbone is a small CNN trained with Mnist data. It allows CenterNet to have a first representation of what a digit is.

All models are saved in the folder: _models_.

### Backbone

A model of the backbone is already trained and available, however you can retrain it using the following command:

```bash
python train_backbone.py [-h] [--name NAME]
```

- _--name_ option allows you to specify the filename to save the model. Default is _backbone_model.pth_.

### CenterNet

Similarly, a trained CenterNet model is also available, however you can retrain it using the following command:

```bash
python train_center_net.py [-h] [--name NAME] [--epochs EPOCHS] [--dataset DATASET] [--batch_size BATCH_SIZE]
```

- _--name_ option allows you to specify the filename to save the model. Default is _center_net_model.pth_
- _--epochs_ option allows you to specify the number of training epochs. Default is 20
- _--dataset_ option allows you to specify the dataset to use for training. Default is _data/mnist_detection_
- _--batch_size_ option allows you to specify the batch size for training. Default is 8

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

Finally, you can test the model on an image using the following command:

```bash
python test_center_net.py [-h] [--nfilter] image
```

- _--nfilter_ option allows you to not filter the bounding boxes.
- _image_ argument is the path to the image to test.

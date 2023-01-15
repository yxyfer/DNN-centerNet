from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
import numpy as np
import math
from .gaussian import gaussian_radius, draw_gaussian


class MnistDetection(Dataset):
    num_classes = 10
    max_objects = 15
    
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    
    train_path = "/train/"
    test_path = "/test/"
    images_path = "/images/"
    labels_path = "/labels/"
    
    gaussian_iou = 0.7
    
    def __init__(self, data_dir: str, train: bool = True, img_shape: tuple = (1, 300, 300)):
        """Create a dataset for the MNIST detection.

        Args:
            data_dir (str): Path to the dataset.
            train (bool, optional): Train dataset?. Defaults to True.
            img_size (tuple, optional): Shape of an Image (C, H, W). Defaults to (1, 300, 300).
        """
        
        self.train = train
        self.img_shape = img_shape
        self.feature_map_size = {
            'h': 75,
            'w': 75,
        }
        
        if train:
            data_dir = data_dir + MnistDetection.train_path
        else:
            data_dir = data_dir + MnistDetection.test_path

        self.images, self.annotations = self._load_data(data_dir)
        self.num_samples = len(self.images)
        
    def _load_data(self, path: str):
        images_path = path + MnistDetection.images_path
        labels_path = path + MnistDetection.labels_path
        
        image_files = os.listdir(images_path)
        image_files = [i for i in image_files if i.endswith(".png")]
        image_files.sort()
        
        label_files = os.listdir(labels_path)
        label_files = [i for i in label_files if i.endswith(".txt")]
        label_files.sort()
        
        images = [cv2.imread(images_path + i, cv2.IMREAD_GRAYSCALE) for i in image_files]
        images = [np.expand_dims(i, axis=2) for i in images]
        
        labels = [pd.read_csv(labels_path + i).values for i in label_files]
        
        return images, labels
    
    def _get_item(self, index: int) -> tuple:
        image = self.images[index]
        labels = np.array(self.annotations[index][:, 0])
        bboxes = np.array(self.annotations[index][:, 1:])
        
        sorted_inds = np.argsort(labels, axis=0)
        bboxes = bboxes[sorted_inds]
        labels = labels[sorted_inds]
        
        return image, labels, bboxes
        
    
    def __getitem__(self, index: int) -> dict:
        image, labels, bboxes = self._get_item(index)
        
        # -- Add resize --
        # -- Add data augmentation --
        
        image = image.astype(np.float32) / 255.
        image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
        
        num_classes = MnistDetection.num_classes
        fmap_size_h = self.feature_map_size['h']
        fmap_size_w = self.feature_map_size['w']
        max_objects = MnistDetection.max_objects
        
        heat_map_tl = np.zeros((num_classes, fmap_size_h, fmap_size_w), dtype=np.float32)
        heat_map_br = np.zeros((num_classes, fmap_size_h, fmap_size_w), dtype=np.float32)
        heat_map_ct = np.zeros((num_classes, fmap_size_h, fmap_size_w), dtype=np.float32)

        # ========= Offset: CornerNet =========
        regs_tl = np.zeros((max_objects, 2), dtype=np.float32)
        regs_br = np.zeros((max_objects, 2), dtype=np.float32)
        regs_ct = np.zeros((max_objects, 2), dtype=np.float32)
        
        inds_tl = np.zeros((max_objects,), dtype=np.int64)
        inds_br = np.zeros((max_objects,), dtype=np.int64)
        inds_ct = np.zeros((max_objects,), dtype=np.int64)
        
        num_objs = np.array(min(bboxes.shape[0], max_objects))
        ind_masks = np.zeros((max_objects,), dtype=np.bool)
        ind_masks[:num_objs] = 1
       
        img_size_h = self.img_shape[1]
        img_size_w = self.img_shape[2]
         
        for i, ((xtl, ytl, xbr, ybr), label) in enumerate(zip(bboxes, labels)):
            xct, yct = (xbr + xtl) / 2., (ybr + ytl) / 2.
            
            fxtl = (xtl * fmap_size_w / img_size_w)
            fytl = (ytl * fmap_size_h / img_size_h)
            fxbr = (xbr * fmap_size_w / img_size_w)
            fybr = (ybr * fmap_size_h / img_size_h)
            fxct = (xct * fmap_size_w / img_size_w)
            fyct = (yct * fmap_size_h / img_size_h)
            
            ixtl = int(fxtl)
            iytl = int(fytl)
            ixbr = int(fxbr)
            iybr = int(fybr)
            ixct = int(fxct)
            iyct = int(fyct)
            
            # Gaussian Heatmap
            width = xbr - xtl
            height = ybr - ytl

            width = math.ceil(width * fmap_size_w / img_size_w)
            height = math.ceil(height * fmap_size_h / img_size_h)
            
            radius = max(0, int(gaussian_radius((height, width), MnistDetection.gaussian_iou)))

            draw_gaussian(heat_map_tl[label], [ixtl, iytl], radius)
            draw_gaussian(heat_map_br[label], [ixbr, iybr], radius)
            draw_gaussian(heat_map_ct[label], [ixct, iyct], radius, delta=5)

            regs_tl[i, :] = [fxtl - ixtl, fytl - iytl]
            regs_br[i, :] = [fxbr - ixbr, fybr - iybr]
            regs_ct[i, :] = [fxct - ixct, fyct - iyct]
            inds_tl[i] = iytl * fmap_size_w + ixtl
            inds_br[i] = iybr * fmap_size_w + ixbr
            inds_ct[i] = iyct * fmap_size_w + ixct
            
            return {
                'image': image,
                'hmap_tl': heat_map_tl, 'hmap_br': heat_map_br, 'hmap_ct': heat_map_ct,
                'regs_tl': regs_tl, 'regs_br': regs_br, 'regs_ct': regs_ct,
                'inds_tl': inds_tl, 'inds_br': inds_br, 'inds_ct': inds_ct,
                'ind_masks': ind_masks
            }
    
    def __len__(self):
        return self.num_samples 
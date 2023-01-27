from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
import numpy as np
import math
from .gaussian import gaussian_radius, draw_gaussian


class MnistDetection(Dataset):
    num_classes = 10
    
    train_path = "/train/"
    test_path = "/test/"
    images_path = "/images/"
    labels_path = "/labels/"
    
    gaussian_iou = 0.6
    
    def __init__(self, data_dir: str, train: bool = True, img_shape: tuple = (1, 300, 300),
                 max_images: int = None, max_objects: int = 30):
        """Create a dataset for the MNIST detection.

        Args:
            data_dir (str): Path to the dataset.
            train (bool, optional): Train dataset?. Defaults to True.
            img_size (tuple, optional): Shape of an Image (C, H, W). Defaults to (1, 300, 300).
            max_images (int, optional): Max number of images to load. Defaults to None = all.
        """
        
        self.train = train
        self.img_shape = img_shape
        self.max_images = max_images
        self.feature_map_size = {
            'h': img_shape[1] // 4,
            'w': img_shape[2] // 4,
        }
        self.max_objects = max_objects
        
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
        
        self.max_images = self.max_images or len(image_files)
        
        image_files = image_files[:self.max_images]
        label_files = label_files[:self.max_images]
        
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

        image, labels, bboxes = self._augment(image, labels, bboxes)

        
        image = image.astype(np.float32) / 255.
        image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
        
        num_classes = MnistDetection.num_classes
        fmap_size_h = self.feature_map_size['h']
        fmap_size_w = self.feature_map_size['w']
        max_objects = self.max_objects
        
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

            draw_gaussian(heat_map_tl[label], (ixtl, iytl), radius)
            draw_gaussian(heat_map_br[label], (ixbr, iybr), radius)
            draw_gaussian(heat_map_ct[label], (ixct, iyct), radius, delta=5)

            regs_tl[i, :] = [fxtl - ixtl, fytl - iytl]
            regs_br[i, :] = [fxbr - ixbr, fybr - iybr]
            regs_ct[i, :] = [fxct - ixct, fyct - iyct]
            inds_tl[i] = iytl * fmap_size_w + ixtl
            inds_br[i] = iybr * fmap_size_w + ixbr
            inds_ct[i] = iyct * fmap_size_w + ixct
        
        bboxes = np.concatenate((bboxes, np.expand_dims(labels, axis=1)), axis=1)
        
        full_bbox = np.zeros((max_objects, 5))
        
        for i, bbox in enumerate(bboxes):
            full_bbox[i] = bbox

        return {
            'image': image,
            'hmap_tl': heat_map_tl, 'hmap_br': heat_map_br, 'hmap_ct': heat_map_ct,
            'regs_tl': regs_tl, 'regs_br': regs_br, 'regs_ct': regs_ct,
            'inds_tl': inds_tl, 'inds_br': inds_br, 'inds_ct': inds_ct,
            'ind_masks': ind_masks, # Number of objects
            'bbox': full_bbox
        }
    
    def __len__(self):
        return self.num_samples

    def _augment(self, image, labels, bboxes):

        def _get_rotation():
            return random.randint(-45,45)

        def _scale(angle, width):
            angle = abs(angle*math.pi/180)
            new_w = width * math.cos(angle) + width*math.sin(angle)
            return width / new_w
        
        def _rotate_xy(x,y,M):
            res = M @ np.array([x,y,1]).T
            return (int(res[1]),int(res[0]))

        def _get_new_box(i, M):
            box = [None, None, None, None]

            #top left
            res = _rotate_xy(i[0], i[1], M)
            dst[res[0], res[1]] = 137
            box[0] = res[1]
            
            #btm right
            res = _rotate_xy(i[2], i[3], M)
            dst[res[0], res[1]] = 137
            box[2] = res[1]
            
            #top right
            res = _rotate_xy(i[2], i[1], M)
            dst[res[0], res[1]] = 137
            box[1] = res[0]
            
            #btm left
            res = _rotate_xy(i[0], i[3], M)
            dst[res[0], res[1]] = 137
            box[3] = res[0]

            return box

        size = image.shape[0]
        angle = _get_rotation()
        coef = _scale(angle, size)


        M = cv2.getRotationMatrix2D(((size-1)/2.0,(size-1)/2.0),angle,coef)
        dst = cv2.warpAffine(image,M,(size,size), cv2.WARP_FILL_OUTLIERS)
        dst = dst.reshape((size,size, 1))
        new_bboxes = []

        for b in bboxes:
            new_bboxes.append(_get_new_box(b,M))
        new_bboxes = np.array(new_bboxes)

        return dst, labels, new_bboxes

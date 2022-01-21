#code library based on the pytorch 1.2

import pandas as pd 
import os
import torch as t 
import numpy as np 
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cfg


class LabelProcessor:

    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256**3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
        return np.array(self.cm2lbl[idx], dtype='int64')    



class LoadDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        """
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        if len(file_path) != 2:
            raise ValueError("unvalid data and label path!")
        self.img_path = file_path[0]
        self.label_path = file_path[1]

        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label = self.center_crop(img, label, self.crop_size)

        img, label = self.img_transform(img, label)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        sample = {'img': img, 'label': label}

        return sample

    def __len__(self):
            return len(self.imgs)


    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, img, label, crop_size):
        img = ff.center_crop(img, crop_size)
        label = ff.center_crop(label, crop_size)
        return img, label

    def img_transform(self, img, label):
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)

        return img, label


label_processor = LabelProcessor(cfg.class_dict_path)
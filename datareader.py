import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

def cointoss(p):
    return random.random() < p

class CustomDataset(Dataset):
    def __init__(self, db_dir, resolution='High', random_crop=None, resize=None, augment_s=True, augment_t=True):
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.db_dir = db_dir  # 直接使用傳入的 db_dir
        
        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        
        self.triplet_list = []
        for folder1 in listdir(self.db_dir):
            folder1_path = join(self.db_dir, folder1)
            if isdir(folder1_path):
                for folder2 in listdir(folder1_path):
                    folder2_path = join(folder1_path, folder2)
                    if isdir(folder2_path):
                        self.triplet_list.append(folder2_path)

        self.triplet_list = np.array(self.triplet_list)
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        folder_path = self.triplet_list[index]
        
        rawFrame0 = Image.open(join(folder_path, "im1.png"))
        rawFrame1 = Image.open(join(folder_path, "im2.png"))
        rawFrame2 = Image.open(join(folder_path, "im3.png"))
        rawFrame3 = Image.open(join(folder_path, "im5.png"))
        rawFrame4 = Image.open(join(folder_path, "im6.png"))
        rawFrame5 = Image.open(join(folder_path, "im7.png"))

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.random_crop)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)
            rawFrame3 = TF.crop(rawFrame3, i, j, h, w)
            rawFrame4 = TF.crop(rawFrame4, i, j, h, w)
            rawFrame5 = TF.crop(rawFrame5, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
                rawFrame3 = TF.hflip(rawFrame3)
                rawFrame4 = TF.hflip(rawFrame4)
                rawFrame5 = TF.hflip(rawFrame5)
            if cointoss(0.5):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)
                rawFrame3 = TF.vflip(rawFrame3)
                rawFrame4 = TF.vflip(rawFrame4)
                rawFrame5 = TF.vflip(rawFrame5)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)
        frame3 = self.transform(rawFrame3)
        frame4 = self.transform(rawFrame4)
        frame5 = self.transform(rawFrame5)

        return frame0, frame1, frame2, frame3, frame4, frame5

    def __len__(self):
        return self.file_len

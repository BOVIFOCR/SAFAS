import os

import PIL.Image
import torch
import pandas as pd
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import math
from glob import glob
import re
from utils.rotate_crop import crop_rotated_rectangle, inside_rect, vis_rotcrop
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt

from ylib.scipy_misc import imread, imsave
from .meta import DEVICE_INFOS

import time
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.seed(0)
SEED = int(1000*time.time()) % 998244353
# SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

ALL_AUG_IS_SPOOF = True
AUG_RATIO = 100


class WFASFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, scale_up=1.1, scale_down=1.0, UUID=-1):
        self.root_dir = root_dir
        self.subset = "dev"
        # self.subset = "test"
        self.video_df = pd.read_csv(
                os.path.join(root_dir, self.subset+".txt"),
                header=None)
        self.is_train = False
        self.transform = transform
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.UUID = UUID
        self.face_width = 400

    def __len__(self):
        return len(self.video_df)

    def __getitem__(self, idx):
        video_path = self.video_df.iloc[idx, 1]

        image_dir = video_path + ('.png' if self.subset == "test" else '.jpg')

        image_x = self.sample_image(image_dir)
        try:
            image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
        except:
            print(image_x.shape, image_dir)
            print(PIL.Image.fromarray(image_x))
            print(self.transform)
            exit(1)
        if self.is_train:
            image_x_view2 = self.transform(PIL.Image.fromarray(image_x))
        else:
            image_x_view2 = image_x_view1

        # só usam de fato image_x_v1, image_x_v2, label, e UUID
        sample = {"image_x_v1": np.array(image_x_view1),
                  "image_x_v2": np.array(image_x_view2),
                  "label": 0,
                  "UUID": self.UUID,
                  'device_tag': 0,
                  'video': image_dir,
                  'client_id': 0,
                  'points': 0}
        return sample

    def sample_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError
        try:
            image = imread(image_path, mode='RGB')
        except OSError:
            print(f'arquivo ruim, {image_path}')
            image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        return image

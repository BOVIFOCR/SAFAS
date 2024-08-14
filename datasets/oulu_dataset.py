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

# TODO return seeds to 0
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


class OuluFaceDataset(Dataset):
    def __init__(self, root_dir, labels_file, is_train, transform=None, scale_up=1.1, scale_down=1.0, UUID=-1):
        self.root_dir = root_dir
        self.video_df = pd.read_csv(labels_file, header=None)
        self.is_train = is_train
        self.transform = transform
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.UUID = UUID
        self.face_width = 400

    def __len__(self):
        return len(self.video_df)

    def __getitem__(self, idx):
        spoofing_label = self.video_df.iloc[idx, 0]
        video_name = self.video_df.iloc[idx, 1]

        no_used_aug = video_name[0] in "0123456789"

        if ALL_AUG_IS_SPOOF:
            spoofing_label &= no_used_aug
        spoofing_label = int(spoofing_label)

        subset_path = "train" if self.is_train else "test"
        image_dir = os.path.join(self.root_dir, subset_path, video_name)

        image_x = self.sample_image(image_dir, video_name)
        image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
        if self.is_train:
            image_x_view2 = self.transform(PIL.Image.fromarray(image_x))
        else:
            image_x_view2 = image_x_view1

        # só usam de fato image_x_v1, image_x_v2, label, e UUID
        sample = {"image_x_v1": np.array(image_x_view1),
                  "image_x_v2": np.array(image_x_view2),
                  "label": spoofing_label,
                  "UUID": self.UUID,
                  'device_tag': 0,
                  'video': image_dir,
                  'client_id': 0,
                  'points': 0}
        return sample

    def sample_image(self, image_dir, video_name):
        frames = glob(os.path.join(image_dir, "*.jpg"))
        frames_total = len(frames)
        if frames_total == 0:
            raise RuntimeError(image_dir)


        for image_id in range(8):
            image_name = f"{video_name}_{image_id}.jpg"
            image_path = os.path.join(image_dir, image_name)

            if os.path.exists(image_path):
                break

        image = imread(image_path)

        return image

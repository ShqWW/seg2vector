from tracemalloc import start
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class BaseTsSet(Dataset):
    def __init__(self, cfg=None, transforms=None):
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.patch_h, self.patch_w = cfg.patch_h, cfg.patch_w
        self.margin_size = cfg.margin_size
        self.patch_num_h, self.patch_num_w = cfg.patch_num_h, cfg.patch_num_w
        start_h_array = np.linspace(start=self.margin_size, stop=self.img_h- (self.patch_h + self.margin_size), num=self.patch_num_h).astype(np.int32)
        start_w_array = np.linspace(start=self.margin_size, stop=self.img_w- (self.patch_w + self.margin_size), num=self.patch_num_w).astype(np.int32)

        self.patch_start_grid = np.array(np.meshgrid(start_w_array, start_h_array)).T.reshape(-1, 2)
        self.patch_num = self.patch_num_h * self.patch_num_w

        self.transforms = transforms

        self.img_path_list = [os.path.join(self.dataset_dir, img_name) for img_name in self.img_name_list]

        self.is_debug = cfg.is_debug

        if self.is_debug:
            self.img_path_list = self.img_path_list[:2]

    def __len__(self):
        return len(self.img_path_list)
    

    def __getitem__(self, index):
        img = cv2.imread(self.img_path_list[index])
        if img.shape[0] < self.img_h or img.shape[1] < self.img_w:
            real_img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            real_img[0:img.shape[0], 0:img.shape[1], :] = img
            img = real_img

        img_name = self.img_name_list[index]

        ori_img = img.copy()

        # prepare img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        patches = self.patch_image_tensor(img)
        item_dict = {
            'imgs': patches,
            'ori_imgs': ori_img,
            'img_name': img_name
        }
        return item_dict

    def patch_image_tensor(self, img):
        patches = []
        for i in range(self.patch_num):
            start_x, start_y = self.patch_start_grid[i, 0], self.patch_start_grid[i, 1]
            patch = img[..., start_y:start_y+self.patch_h, start_x:start_x+self.patch_w]
            patches.append(patch.unsqueeze(0))
        patches = torch.cat(patches, dim=0)
        return patches
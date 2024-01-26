import numpy as np
import torch
import os
from PIL import Image


class KidneyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transforms=None):
        if split == "train":
            self.imgs_dir = os.path.join(data_dir, "kidney_1_dense", "images")
            self.msks_dir = os.path.join(data_dir, "kidney_1_dense", "labels")
        elif split == "validation":
            self.imgs_dir = os.path.join(data_dir, "kidney_1_sparse", "images")
            self.msks_dir = os.path.join(data_dir, "kidney_1_sparse", "labels")

        self.slices_ids = sorted(os.listdir(self.imgs_dir))
        self.transforms = transforms
        self.h = Image.open(os.path.join(self.imgs_dir, self.slices_ids[0])).height
        self.w = Image.open(os.path.join(self.imgs_dir, self.slices_ids[0])).width

    def __len__(self):
        return len(self.slices_ids)

    def __getitem__(self, idx):
        slice_id = self.slices_ids[idx]
        img_path = os.path.join(self.imgs_dir, slice_id)
        msk_path = os.path.join(self.msks_dir, slice_id)

        img = Image.open(img_path)
        msk = Image.open(msk_path)
        img = np.array(img, dtype=np.float32)
        msk = np.array(msk)

        if self.transforms is not None:
            t = self.transforms(image=img, mask=msk)
            img = t["image"]
            msk = t["mask"]

        img = torch.from_numpy(img)[None, :]
        msk = torch.as_tensor(msk)
        img /= 31000
        msk = msk // 255

        return img, msk.float()

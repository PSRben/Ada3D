import torch
import h5py
import cv2
import numpy as np
import torch.utils.data as data


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)

        self.gt = data.get("GT")
        self.lms = data.get("LMS")
        self.ms = data.get("MS")
        self.pan = data.get("PAN")

    def __getitem__(self, index):
        return torch.from_numpy(self.gt[index, :, :, :]).float(), \
               torch.from_numpy(self.lms[index, :, :, :]).float(), \
               torch.from_numpy(self.ms[index, :, :, :]).float(), \
               torch.from_numpy(self.pan[index, :, :, :]).float()

    def __len__(self):
        return self.gt.shape[0]


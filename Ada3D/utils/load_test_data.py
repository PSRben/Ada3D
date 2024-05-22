import cv2
import h5py
import torch
import numpy as np
import scipy.io as sio


def load_h5py(file_path):
    data = h5py.File(file_path)
    
    ms = data["MS"][...]
    ms = np.array(ms, dtype=np.float32)
    ms = torch.from_numpy(ms)

    lms = data["LMS"][...]
    lms = np.array(lms, dtype=np.float32)
    lms = torch.from_numpy(lms)

    pan = data["PAN"][...]
    pan = np.array(pan, dtype=np.float32)
    pan = torch.from_numpy(pan)

    return ms, lms, pan

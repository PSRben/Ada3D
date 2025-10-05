# Adaptive 3D Convolution for Remote Sensing Image Fusion (Ada3D)


# Get Started
## Datasets
- Datasets for pansharpening: [PanCollection](https://github.com/liangjiandeng/PanCollection). 
We recommend downloading the dataset in h5py format.

- Datasets for hyper-spectral pansharpening: [HyperPanCollection](https://github.com/liangjiandeng/HyperPanCollection).
We recommend downloading the dataset in h5py format.

- Dataset for HISR: the CAVE dataset. You can find this dataset on the Internet.

## Installation
This project is suitable for all versions of PyTorch after 1.7.1. Besides, you need to install some other packages as below:
```
pip install einops h5py opencv-python torchinfo scipy numpy
```

## Usage
- This repository is only for the hyper-spectral pansharpening task.

- The model weights can be found in the weights dir.

- Training and testing commands (with the WDC Dataset):

```
# train
python train.py --train_data_path ./path_to_data/Train_WDC.h5 --val_data_path ./path_to_data/Valid_WDC.h5
# test
python test.py --file_path ./path_to_data/name.h5 --save_dir ./path_to_dir --weight ./weights/hspansharpening/WDC/1200.pth
```



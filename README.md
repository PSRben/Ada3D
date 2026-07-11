# Ada3D
- Code for the paper: "Adaptive 3D Convolution for Remote Sensing Image Fusion", IEEE TIP, 2026.

- First application of adaptive 3D convolution in image fusion.

- State-of-the-art (SOTA) performance in pansharpening, hyper-spectral pansharpening, and HISR tasks.

# Paper
- For a detailed understanding of our method, please refer to the paper: [Adaptive 3D Convolution for Remote Sensing Image Fusion](https://arxiv.org/abs/2605.09455).
- This paper has been published in the IEEE Transactions on Image Processing (TIP).

# Get Started
## Datasets
- Datasets for pansharpening: [PanCollection](https://github.com/liangjiandeng/PanCollection). 
We recommend downloading the dataset in h5py format.

- Datasets for hyper-spectral pansharpening: [HyperPanCollection](https://github.com/liangjiandeng/HyperPanCollection).
We recommend downloading the dataset in h5py format.

- Dataset for HISR: the CAVE dataset. You can find this dataset on the Internet.

- Results for Ada3D and a series of methods on PanCollection/HyperPanCollection/CAVE: https://pan.baidu.com/s/1ARGLyvGKn57-eCl041Gk3g, key: 6271.

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

# Citation
```
@ARTICLE{11513694,
  author={Peng, Siran and Zhu, Xiangyu and Deng, Shang-Qi and Deng, Liang-Jian and Lei, Zhen},
  journal={IEEE Transactions on Image Processing}, 
  title={Adaptive 3D Convolution for Remote Sensing Image Fusion}, 
  year={2026},
  volume={35},
  number={},
  pages={4975-4988},
  doi={10.1109/TIP.2026.3689418}}
```

# Contact
We are glad to hear from you. If you have any questions, please feel free to contact siran_peng@163.com.


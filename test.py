import os
import torch
import argparse
import numpy as np
import scipy.io as sio
from utils.load_test_data import load_h5py
from model.hspansharpening.convnet3d import ConvNet3D as ConvNet3D


def test(args):
    # load data
    model = ConvNet3D(device=args.device).to(args.device).eval()
    weight = torch.load(args.weight)

    try: 
        model.load_state_dict(weight)
    except:
        new_weight = {}
        for k, v in weight.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            new_weight[new_k] = v
        model.load_state_dict(new_weight)

    ms, _, pan = load_h5py(args.file_path)
    print('Data loaded.')

    # get size
    image_num, C, h, w = ms.shape
    _, _, H, W = pan.shape
    cut_size = 128  # must be divided by 4, we recommand 64
    ms_size = cut_size // 4
    pad = 0  # must be divided by 4
    edge_H = (cut_size - (H - (H // cut_size) * cut_size)) % cut_size
    edge_W = (cut_size - (W - (W // cut_size) * cut_size)) % cut_size
    
    OUTPUT = torch.zeros(image_num, C, H, W).to(args.device)

    #################### test code ###################
    for k in range(image_num):
        print('Processing the {}th image...'.format(k+1))
        with torch.no_grad():
            x1, x2 = ms[k, :, :, :], pan[k, 0, :, :]
            x1 = x1.unsqueeze(dim=0).to(args.device)
            x2 = x2.unsqueeze(dim=0).unsqueeze(dim=1).to(args.device)
            x1_pad = torch.zeros(1, C, h + pad // 2 + edge_H // 4, w + pad // 2 + edge_W // 4).to(args.device)
            x2_pad = torch.zeros(1, 1, H + pad * 2 + edge_H, W + pad * 2 + edge_W).to(args.device)
            x1 = torch.nn.functional.pad(x1, (pad // 4, pad // 4, pad // 4, pad // 4), 'reflect')
            x2 = torch.nn.functional.pad(x2, (pad, pad, pad, pad), 'reflect')
            x1_pad[:, :, :h + pad // 2, :w + pad // 2] = x1
            x2_pad[:, :, :H + pad * 2, :W + pad * 2] = x2
            output = torch.zeros(1, C, H + edge_H, W + edge_W).to(args.device)
            scale_H = (H + edge_H) // cut_size
            scale_W = (W + edge_W) // cut_size
            for i in range(scale_H):
                for j in range(scale_W):
                    MS = x1_pad[:, :, i * ms_size: (i + 1) * ms_size + pad // 2,
                         j * ms_size: (j + 1) * ms_size + pad // 2]
                    PAN = x2_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                          j * cut_size: (j + 1) * cut_size + 2 * pad]
                    sr = model(MS, PAN)
                    sr = torch.clamp(sr, 0, 1)
                    output[:, :, i * cut_size: (i + 1) * cut_size, j * cut_size: (j + 1) * cut_size] = \
                        sr[:, :, pad: cut_size + pad, pad: cut_size + pad]
            output = output[:, :, :H, :W]
            output = torch.squeeze(output)
            OUTPUT[k] = output
    #########################################################
    
    OUTPUT = OUTPUT.permute(0, 2, 3, 1).cpu().detach().numpy()  # HxWxC
    save_name = os.path.join(args.save_dir, "Ada3D.mat")
    sio.savemat(save_name, {'output': OUTPUT})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='', help='Absolute path of the test file (in h5 format).')
    parser.add_argument('--save_dir', type=str, default='', help='Absolute path of the save dir.')
    parser.add_argument('--weight', type=str, default='', help='Path of the weight.')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    test(args)

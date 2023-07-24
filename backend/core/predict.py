import os
import sys
import cv2
import torch
import core.net.unet as net
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

import os

rate = 0.5

import os
import nibabel as nib
import numpy as np
import cv2

def predict(nii_path):
    img = nib.load(nii_path)
    data = img.get_fdata()
    slices = []
    for i in range(data.shape[2]):
        slice_data = np.rot90(data[:, :, i])
        slices.append(slice_data.tolist())
    predict_img = np.array(slices[0])
    # 保存为图片
    predict_path = './tmp/predict/' + nii_path.split('/')[-1].split('.')[0] + '.png'
    print("======predict_path",predict_path)
    cv2.imwrite(predict_path, predict_img)
    return predict_path, slices



if __name__ == '__main__':
    # 写保存模型
    # train()
    predict()

## Created by lucas Rouhier On january 4,2020. Please Cite :
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import numpy as np
from skimage.morphology import square
from skimage.morphology import dilation
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle
import nibabel as nib
import math
import sys

sys.path.insert(0, '/home/GRAMES.POLYMTL.CA/luroub/luroub_local/lurou_local/sct/sct/')
#import spinalcordtoolbox.image as Image
matplotlib.use("Agg")


def label2MaskMap(data, c_dx=0, c_dy=0, radius=10, normalize=False):
    """
    Generate a Mask map from the coordenates
    :param M, N: dimesion of output
    :param position: position of the label
    :param radius: is the radius of the gaussian function
    :return: a MxN normalized array
    """

    # Our 2-dimensional distribution will be over variables X and Y
    (M, N) = (300, 200)
    if len(data) <= 2:
        data = [data]

    maskMap = []
    for index, value in enumerate(data):
        x, y = value

        # Correct the labels
        x = x + c_dx
        y = y + c_dy

        X = np.linspace(0, M - 1, M)
        Y = np.linspace(0, N - 1, N)
        X, Y = np.meshgrid(X, Y)
        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        # Mean vector and covariance matrix
        mu = np.array([x, y])
        Sigma = np.array([[radius, 0], [0, radius]])

        # The distribution on the variables X, Y packed into pos.
        Z = multivariate_gaussian(pos, mu, Sigma)

        # Normalization
        if normalize:
            Z = Z * (1 / np.max(Z))
        else:
            # 8bit image values (the loss go to inf+)
            Z = Z * (1 / np.max(Z))
            Z = np.asarray(Z * 255, dtype=np.uint8)

        maskMap.append(Z)

    if len(maskMap) == 1:
        maskMap = maskMap[0]

    return np.asarray(maskMap)


def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian distribution on array.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def add_zero_padding(img_list, x_val=512, y_val=512):
    if type(img_list) != list:
        img_list = [img_list]
    img_zero_padding_list = []
    for i in range(len(img_list)):
        print(i)
        # print('Doing zero-padding ' + str(i + 1) + '/' + str(len(img_list)))
        img = img_list[i]
        img_tmp = np.zeros((x_val, y_val, 1), dtype=np.float64)
        img_tmp[0:img.shape[0], 0:img.shape[1], 0] = img
        img_zero_padding_list.append(img_tmp)

    return img_zero_padding_list


def mask2label(path_label):
    a = nib.load(path_label)
    arr = np.array(a.dataobj)

    list_label_image = []
    for i in range(len(arr.nonzero()[0])):
        x = arr.nonzero()[0][i]
        y = arr.nonzero()[1][i]
        z = arr.nonzero()[2][i]
        list_label_image.append([x, y, z, arr[x, y, z]])
    list_label_image.sort(key=lambda x: x[3])
    return (list_label_image)


def get_midNifti(path_im, ind):
    a = nib.load(path_im)
    arr = np.array(a.dataobj)
    return np.mean(arr[ind - 3:ind + 3, :, :], 0)


def images_normalization(img_list, std=True):
    if type(img_list) != list:
        img_list = [img_list]
    img_norm_list = []
    for i in range(len(img_list)):
        # print('Normalizing ' + str(i + 1) + '/' + str(len(img_list)))
        img = img_list[i] - np.mean(img_list[i])  # zero-center
        if std:
            img_std = np.std(img)  # normalize
            epsilon = 1e-100
            img = img / (img_std + epsilon)  # epsilon is used in order to avoid by zero division
        img_norm_list.append(img)
    return img_norm_list


def load_Data_Bids2Array(DataSet_path, mode=0):
    # Mode 1 only load T1 , Mode 2 only load T2 , Different number load both
    size_val = 512
    ds_image = []
    ds_label = []
    list_dir = os.listdir(DataSet_path)
    list_dir.sort()
    if '.DS_Store' in list_dir:
        list_dir.remove('.DS_Store')
    a = len(list_dir)
    for i in range(5):
        path_tmp = DataSet_path + list_dir[i] + '/'
        if mode != 2:
            tmp_label = mask2label(path_tmp + 'T1_label-disc-manual_straight.nii.gz')
        if mode != 1:
            tmp_label_t2 = mask2label(path_tmp + 'T2_label-disc-manual_straight.nii.gz')

        if mode != 1:
            index_mid = tmp_label_t2[0][0]
        else:
            index_mid = tmp_label[0][0]
        if mode != 2:
            mid_slice = get_midNifti(path_tmp + 'T1w_straight.nii.gz', index_mid)
        if mode != 1:
            mid_slice_t2 = get_midNifti(path_tmp + 'T2W_straight.nii.gz', index_mid)
        if mode == 2:
            mid_slice = mid_slice_t2
        if mid_slice.shape[0] > 200:
            print('removed')
            pass
        elif mid_slice.shape[1] > 200:
            print('removed')
            pass
        else:
            if mode != 2:
                ds_image.append(mid_slice)
                ds_label.append(tmp_label)
            if mode != 1:
                ds_image.append(mid_slice_t2)
                ds_label.append(tmp_label_t2)

    ds_image = images_normalization(ds_image)

    # Zero padding
    ds_image = add_zero_padding(ds_image, x_val=150, y_val=200)
    # val_ds_img = add_zero_padding(val_ds_img, x_val=size_val, y_val=size_val)
    # test_ds_img = add_zero_padding(test_ds_img, x_val=size_val, y_val=size_val)

    # Convert images to np.array
    ds_image = np.array(ds_image)

    return [ds_image, ds_label]

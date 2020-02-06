# Author: Lucas
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file license.md
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from Data2array import *
import matplotlib.pyplot as plt
import PIL

# normalize Image
def normalize(arr):
    ma = arr.max()
    mi = arr.min()
    return ((arr - mi) / (ma - mi))


# Useful function to generate a Gaussian Function on given coordinates. Used to generate groudtruth.
def label2MaskMap_GT(data, shape, c_dx=0, c_dy=0, radius=10, normalize=False):
    """
    Generate a Mask map from the coordenates
    :param shape: dimension of output
    :param data : input image
    :param radius: is the radius of the gaussian function
    :param normalize : bool for normalization.
    :return: a MxN normalized array
    """

    # Our 2-dimensional distribution will be over variables X and Y
    (M, N) = (shape[2], shape[1])
    if len(data) <= 2:
        # Output coordinates are reduced during post processing which poses a problem
        data = [0, data[0], data[1]]
    maskMap = []

    x, y = data[2], data[1]

    # Correct the labels
    x += c_dx
    y += c_dy

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
        Z *= (1 / np.max(Z))
    else:
        # 8bit image values (the loss go to inf+)
        Z *= (1 / np.max(Z))
        Z = np.asarray(Z * 255, dtype=np.uint8)

    maskMap.append(Z)

    if len(maskMap) == 1:
        maskMap = maskMap[0]

    return np.asarray(maskMap)


def extract_all(list_coord_label, shape_im=(1, 150, 200)):
    """
    Create groundtruth by creating gaussian Function for every ground truth points for a single image
    :param list_coord_label: list of ground truth coordinates
    :param shape_im: shape of output image with zero padding
    :return: a 2d heatmap image.
    """
    final = np.zeros(shape_im)
    for x in list_coord_label:
        train_lbs_tmp_mask = label2MaskMap_GT(x, shape_im)
        for w in range(shape_im[1]):
            for h in range(shape_im[2]):
                final[0, w, h] = max(final[0, w, h], train_lbs_tmp_mask[w, h])
    return (final)


def extract_groundtruth_heatmap(DataSet):
    """
    Loop across images to create the dataset of groundtruth and images to input for training
    :param DataSet: An array containing [images, GT corrdinates]
    :return: an array containing [image, heatmap]
    """
    [train_ds_img, train_ds_label] = DataSet

    global testing_image
    testing_image = train_ds_img[-1]
    tmp_train_labels = [0 for i in range(len(train_ds_label))]
    tmp_train_img = [0 for i in range(len(train_ds_label))]
    train_ds_img = np.array(train_ds_img)

    for i in range(len(train_ds_label)):
        final = extract_all(train_ds_label[i])
        tmp_train_labels[i] = normalize(final[0, :, :])

    tmp_train_labels = np.array(tmp_train_labels)

    for i in range(len(train_ds_img)):
        tmp_train_img[i] = (normalize(train_ds_img[i][:, :, 0]))

    tmp_train_labels = np.expand_dims(tmp_train_labels, axis=-1)
    tmp_train_img = np.expand_dims(train_ds_img, axis=-1)
    return [tmp_train_img, tmp_train_labels]


class image_Dataset(Dataset):
    def __init__(self, image_paths, target_paths):  # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(20,  PIL.Image.BILINEAR),transforms.RandomCrop(120),transforms.RandomVerticalFlip(0.6),transforms.ToTensor()])
        

    def __getitem__(self, index):
        mask = self.target_paths[index]
        mask = mask.astype(np.float32) 

        image = self.image_paths[index]
        image = image.astype(np.float32)

        t_image = self.transform(image)
        t_mask = self.transform(mask)

        return t_image, t_mask

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)

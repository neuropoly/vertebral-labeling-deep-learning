from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from Data2array import *
import matplotlib.pyplot as plt


def normalize(arr):
    ma = arr.max()
    mi = arr.min()

    return ((arr - mi) / (ma - mi))
def label2MaskMap_2(data,shape, c_dx = 0, c_dy = 0, radius = 10, normalize = False):
    """
    Generate a Mask map from the coordenates
    :param M, N: dimesion of output
    :param position: position of the label
    :param radius: is the radius of the gaussian function
    :return: a MxN normalized array
    """

    # Our 2-dimensional distribution will be over variables X and Y
    (M,N)=(shape[2],shape[1])
    if len(data)<=2:
        data = [data]

    maskMap = []
    for index, value in enumerate(data):
        if len(value)>2:
            print('value',value)
            value=[value[2],value[1]]

        x,y = value

        #Correct the labels
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

def label2MaskMap_GT(data, shape, c_dx=0, c_dy=0, radius=10, normalize=False):
    """
    Generate a Mask map from the coordenates
    :param M, N: dimension of output
    :param position: position of the label
    :param radius: is the radius of the gaussian function
    :return: a MxN normalized array
    """

    # Our 2-dimensional distribution will be over variables X and Y
    (M, N) = (shape[2], shape[1])
    if len(data) <= 2:
        data = [0,data[0],data[1]]
    #print(data)
    #print(np.transpose(data.shape))

    maskMap = []

    x, y = data[2], data[1]

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


def extract_all(list_coord_label, shape_im=(1, 150, 200)):
    final = np.zeros(shape_im)
    final_2=np.zeros(shape_im)
    print('list given to extract all',list_coord_label)
    for x in list_coord_label:
        print('x',x)
        train_lbs_tmp_mask = label2MaskMap_GT(x, shape_im)
        for w in range(shape_im[1]):
            for h in range(shape_im[2]):
                final[0, w, h] = max(final[0, w, h], train_lbs_tmp_mask[w, h])
    plt.imshow(final[0,:,:])
    plt.savefig('newbutsmart.png')
    plt.imshow(final_2[0,:,:])
    plt.savefig('newprop.png')
    return (final)


def extract_groundtruth_heatmap(DataSet):
    [train_ds_img, train_ds_label] = DataSet

    global testing_image
    testing_image = train_ds_img[-1]

    epochs = 30
    batch_size = 10

    # Selecting only the train images that contatins the end label
    # in these case 15 --> from C2-C3 to T7-T8 disc
    # remember that with these method big amount of training images will be descarted

    tmp_train_labels = [0 for i in range(len(train_ds_label))]
    tmp_train_img = [0 for i in range(len(train_ds_label))]

    train_ds_img = np.array(train_ds_img)

    # print(len(train_ds_label))
    # print(train_lbs_tmp.shape)
    for i in range(len(train_ds_label)):
        final = extract_all(train_ds_label[i])
        tmp_train_labels[i] = normalize(final[0, :, :])

        # Now make the trining datases using only the subset with that contain the "label_number" label
    tmp_train_labels = np.array(tmp_train_labels)
    for i in range(len(train_ds_img)):
        tmp_train_img[i] = (normalize(train_ds_img[i][:, :, 0]))
    # for i in range (len(tmp_train_labels)):
    # tmp_train_labels[i]=tmp_train_labels[i]>0.3
    tmp_train_labels = np.expand_dims(tmp_train_labels, axis=-1)

    tmp_train_img = np.expand_dims(train_ds_img, axis=-1)
    return [tmp_train_img, tmp_train_labels]


class valdataset(Dataset):
    def __init__(self, image_paths, target_paths):  # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        mask = self.target_paths[index]

        image = self.image_paths[index]

        t_image = self.transforms(image)
        t_mask = self.transforms(mask)

        return t_image, t_mask

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)

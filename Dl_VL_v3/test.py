from Metrics import *
import torch
import numpy as np
from train_utils import *
from Data2array import *
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import model as m


def prediction_coordinates(Image):
    shape_im = Image.shape
    shape_im = sorted(shape_im)
    final, coordinates = infer_image(Image)
    #print(coordinates)
    final = np.zeros(shape_im)
    print('before post_proc')
    for x in coordinates:

        train_lbs_tmp_mask = label2MaskMap_2(x, shape_im)
        # plt.imshow(train_lbs_tmp_mask)
        # plt.show()
        for w in range(shape_im[1]):
            for h in range(shape_im[2]):
                final[0, w, h] = max(final[0, w, h], train_lbs_tmp_mask[w, h])
    # plt.imshow(np.add(final[0, :, :] * 1, 40 * Image[:, :, 0]))
    # plt.show()
    print('post_processing')
    final = np.zeros(shape_im)
    coord_out = post_processing(coordinates)
    print('calculating metrics on image')
    print(coord_out)
    mesure_err_disc(coord_gt[i], coord_out, distance_l2)
    mesure_err_z(coord_gt[i], coord_out, zdis)
    fp = Faux_pos(coord_gt[i], coord_out,tot)
    fn = Faux_neg(coord_gt[i], coord_out)
    faux_pos.append(fp)
    faux_neg.append(fn)
    for x in coord_out:
        print('x',x)

        train_lbs_tmp_mask = label2MaskMap_GT(x, shape_im)
        for w in range(shape_im[1]):
            for h in range(shape_im[2]):
                final[0, w, h] = max(final[0, w, h], train_lbs_tmp_mask[w, h])
    # plt.imshow(np.add(final[0, :, :, ] * 1, 40 * Image[:, :, 0]))
    # plt.show()


def post_processing(coordinates):
    c = 0
    coordinates_tmp = []
    # print(coordinates)
    coordinates = sorted(coordinates, key=lambda x: x[0])
    width_pos = [x[1] for x in coordinates]
    height_pos = [x[0] for x in coordinates]

    mean = np.median(width_pos)
    to_remove = []
    for i in range(len(width_pos)):
        if abs(width_pos[i] - mean) > 15:
            to_remove.append(i)
    for i in range(len(coordinates)):
        if i in to_remove:
            pass
        else:
            coordinates_tmp.append(coordinates[i])

    width_pos_c = [x[1] for x in coordinates_tmp]
    height_pos_c = [x[0] for x in coordinates_tmp]
    new_height = []
    new_width = []
    i = 0
    while i < len(height_pos_c):
        j = i
        tmp = []

        while j < len(height_pos_c) - 2 and abs(height_pos_c[j] - height_pos_c[j + 1]) < 10:
            if len(tmp) > 0:
                if abs(height_pos_c[tmp[0]] - height_pos_c[j + 1]) > 20:
                    # print('toomuch')
                    break
                #else:
            # print(height_pos_c[tmp[0]] - height_pos_c[j + 1])
            tmp.append(j)
            j = j + 1
           # print(tmp)

        if len(tmp) > 0:

            h = np.round(np.mean(height_pos_c[tmp[0]:tmp[len(tmp) - 1] + 1]))
            w = np.round(np.mean(width_pos_c[tmp[0]:tmp[len(tmp) - 1] + 1]))

            new_height.append(h)
            new_width.append(w)
            i = i + len(tmp) + 1

        else:
            h = height_pos_c[i]
            w = width_pos_c[i]

            new_height.append(h)
            new_width.append(w)
            i = i + 1
    coordinates_tmp = []
    for a in range(len(new_height)):
        coordinates_tmp.append([new_height[a], new_width[a]])
    coordinates = sorted(coordinates_tmp, key=lambda x: x[0])
    width_pos_c = [x[1] for x in coordinates_tmp]
    height_pos_c = [x[0] for x in coordinates_tmp]
    distance = []
    coord_out = []

    to_remove = []
    for i in range(len(height_pos_c) - 1):
        distance.append(height_pos_c[i + 1] - height_pos_c[i])
    dis_mean_g = np.mean(distance)
    # print(dis_mean)
    for i in range(1, len(height_pos_c) - 1):
        dis_mean = dis_mean_g

       # print(height_pos_c[i + 1] - height_pos_c[i])
        if i < len(height_pos_c) - 4:
            distance_2 = []
            for j in range(4):
                distance_2.append(height_pos_c[i + j] - height_pos_c[i + j - 1])
                dis_mean = np.median(distance_2)
            print(dis_mean)
        # if abs(abs((height_pos_c[i+1]-height_pos_c[i]))-abs((height_pos_c[i]-height_pos_c[i-1])))>10 and abs(abs((height_pos_c[i+1]-height_pos_c[i]))-abs((height_pos_c[i]-height_pos_c[i-1])))<35:
        if abs(height_pos_c[i] - height_pos_c[i - 1]) < dis_mean - 0.1 * dis_mean:
            if abs(height_pos_c[i + 1] - height_pos_c[i]) < dis_mean - 0.7 * dis_mean:
                to_remove.append(i)
                # print(coordinates_tmp[i])
            # print('dif dis')
            # print(abs(abs((height_pos_c[i + 1] - height_pos_c[i])) - abs((height_pos_c[i] - height_pos_c[i - 1]))))
    for i in range(len(coordinates_tmp)):
        if i in to_remove:
            pass
        else:
            coord_out.append(coordinates_tmp[i])

    return (coord_out)


def retrieves_gt_coord(ds):
    coord_gt = []
    for i in range(len(ds[1])):
        coord_tmp = [[], []]
        for j in range(len(ds[1][i])):
            if ds[1][i][j][3] == 1 or ds[1][i][j][3] > 30:
                pass
            else:
                coord_tmp[0].append(ds[1][i][j][2])
                coord_tmp[1].append(ds[1][i][j][1])
        coord_gt.append(coord_tmp)
    return (coord_gt)
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


def infer_image(image, c=0.02):
    coord_out = []
    shape_im = image.shape
    final = np.zeros((shape_im[0], shape_im[1]))
    shape_im = sorted(shape_im)
    # retrieve coordinates of extreme point
    # coordinates.sort()
    # originx=coordinates[-1][0]
    # originx=6
    originy = 0
    # image=normalize(image[:,:,0])

    # print(originy)
    patch = image[:, :, 0]
    patch = normalize(patch)
    # patch = skimage.exposure.equalize_adapthist(patch,kernel_size=20,clip_limit=0.02)

    patch = np.expand_dims(patch, axis=-1)
    # patch=Ying_2017_CAIP(patch)

    patch = transforms.ToTensor()(patch).unsqueeze(0).cuda()
    patch = patch.double()
    patch_out = model(patch)
    patch_out = patch_out.data.cpu().numpy()
   # plt.imshow(patch_out[0, 0, :, :])
   # plt.show()
    coordinates_tmp = peak_local_max(patch_out[0, 0, :, :], min_distance=5, threshold_abs=0.5)
    for w in range(patch.shape[0]):
        for h in range(patch.shape[1]):
            final[w, h] = max(final[w, h], patch_out[0, 0, w, h])
        # final[0,0,originx-5:originx+80,originy-25:originy+50]=patch_out[0,0,:,:]

    for x in coordinates_tmp:
        coord_out.append([x[1], x[0]])
        # coord_out.append(coordinates_tmp)
        # print(originy)
    # print(coord_out)
    return (final, coord_out)

print('load image')
path = '/home/GRAMES.POLYMTL.CA/luroub/luroub_local/lurou_local/deep_VL_2019/straight/'
ds = load_Data_Bids2Array(path, mode=2)
print('extract mid slices')
full = extract_groundtruth_heatmap(ds)
full[0] = full[0][:,:,:,:,0]
print('retrieving ground truth coordinates')
coord_gt = retrieves_gt_coord(ds)
# intialize metrics
distance_l2 = []
zdis = []
faux_pos = []
faux_neg = []
compteur = []
compteur_tot = []
tot = []

model = m.ModelCountception_v2(inplanes=1, outplanes=1)
model = model.cuda()
model = model.double()
model.load_state_dict(torch.load("checkpoints/Countception_L1run.model")['model_weights'])
for i in range(len(coord_gt)):
    #print(i)
    # path_tmp=path+x
    # mid_check=load_Data_just_check(path_tmp)
    prediction_coordinates(full[0][i][:, :, :])
    #print(coord_gt[i])
    print('processing image {:d} out of {:d}'.format(i+1,len(coord_gt)))

print('distance med l2 and std ' + str(np.median(distance_l2)))
print(np.std(distance_l2))
print('distance med z and std ' + str(np.mean(zdis)))
print(np.std(zdis))
print('faux neg per image ',faux_neg)
print('total number of points ' + str(np.sum(tot)))
print('number of faux neg ' + str(np.sum(faux_neg)))
print('number of faux pos ' + str(np.sum(faux_pos)))
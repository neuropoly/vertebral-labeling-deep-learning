import numpy as np


def closest_node(node, nodes):
    nodes1 = np.asarray(nodes)
   # print(nodes)
    dist_2 = np.sum((nodes1 - node) ** 2, axis=1)
    return np.argmin(dist_2)


def mesure_err_disc(gt, pred, dis):
    loss = []
    #print(pred)
    for i in range(len(gt[0])):
        node = np.array([gt[0][i], gt[1][i]])
        h = closest_node(node, pred)
        dis.append(np.linalg.norm(node - pred[h]))


def mesure_err_z(gt, pred, z):
    for i in range(len(gt[0])):
        node = np.array([gt[0][i], gt[1][i]])
        h = closest_node(node, pred)
        z.append(abs(node[0] - pred[h][0]))


def Faux_pos(gt, pred,tot):
    c = 0

    gt = np.transpose(gt)
    tot.append(len(gt))
    already_used = []
    for i in range(len(pred)):

        node = np.array([pred[i][0], pred[i][1]])
        h = closest_node(node, gt)
        if (abs(node[1] - gt[h][1])) > 10:
            print('fauxP')
            c = c + 1
        elif h in already_used:
            c = c + 1
    return c


def Faux_neg(gt, pred):
    c = 0
    gt = np.transpose(gt)
    for i in range(len(gt[0])):
        node = np.array([gt[0][i], gt[1][i]])
        h = closest_node(node, pred)
        if (abs(node[0] - pred[h][0])) > 5:
            c = c + 1
    return c

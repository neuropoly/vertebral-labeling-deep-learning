from Data2array import *
from train_utils import *
import model as m
from losses import *
import numpy as np
import copy
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

path = '/home/GRAMES.POLYMTL.CA/luroub/luroub_local/lurou_local/deep_VL_2019/straight/'
print('load dataset')
ds = load_Data_Bids2Array(path, mode=2)
print('creating heatmap')
full = extract_groundtruth_heatmap(ds)

# put it in the torch loader

train_idx = int(np.round(len(full[0]) * 0.6))
validation_idx = int(np.round(len(full[0]) * 0.85))
print(full[0].shape)
full[0]=full[0][:,:,:,:,0]
full_dataset_train = valdataset(image_paths=full[0][0:train_idx], target_paths=full[1][:train_idx])
full_dataset_val = valdataset(image_paths=full[0][train_idx:validation_idx],
                              target_paths=full[1][train_idx:validation_idx])

print('lendata' + str(len(full_dataset_train)))
train_loader = DataLoader(full_dataset_train, batch_size=2,
                          shuffle=False,
                          num_workers=0)
val_loader = DataLoader(full_dataset_val, batch_size=2,
                        shuffle=False,
                        num_workers=0)
print('generating model')
model = m.ModelCountception_v2(inplanes=1, outplanes=1)
model = model.cuda()
model = model.double()

criterion = caffe_eucl_loss
solver = optim.Adam(model.parameters(), lr=0.00005)
loss_fcd = FocalDiceLoss()
print('training')
for epoch in range(1000):
    for idx, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        output = model.forward(input)
        heat = output.data.cpu().numpy()
        # plt.imshow(heat[0,0,:,:])
        # plt.show()
        loss = criterion(output, target)
        loss_wing = AdapWingLoss(output, target)
        loss_2 = dice_loss(output, target)

        # Zero grad
        model.zero_grad()
        loss.backward(retain_graph=True)
        loss_wing.backward(retain_graph=True)
        loss_2.backward()
        solver.step()

    # print(loss_l2_disk(output, target))
    with torch.no_grad():
        print('val_mode')
        val_loss = []
        for idx, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model.forward(input)
            if (epoch + 1) % 1 == 0:
                heat = output.data.cpu().numpy()
                plt.imshow(heat[0, 0, :, :] > 0.5)
                plt.show()
            val_loss.append(criterion(output, target).item())

        print("Epoch", epoch, "- Validation Loss:", np.mean(val_loss))

    if (epoch + 1) % 100 == 0:
        state = {'model_weights': model.state_dict()}
        torch.save(state, "checkpoints/Countception_vetebral_labeling.model".format(epoch))

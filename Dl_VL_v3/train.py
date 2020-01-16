# Author: Lucas
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file license.md

from Data2array import *
from train_utils import *
import model as m
from losses import *
import numpy as np
import copy
import torch
import yaml
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils


def main():
    cuda_available = torch.cuda.is_available()
    # import configuration
    conf = yaml.load(open('config.yml'))
    # Path to data is stored in config file
    path = conf['path_to_data']
    print('load dataset')
    ds = load_Data_Bids2Array(path, mode=conf['mode'], split='train')
    print('creating heatmap')
    full = extract_groundtruth_heatmap(ds)

    # put it in the torch loader
    # The 60 first pourcent are for training the 25 next are for validation in an attempt to keep the 15 last for test
    # TO DO : Implement something to load only this 85 % and not all
    train_idx = int(np.round(len(full[0]) * 0.75))
    validation_idx = int(np.round(len(full[0])))
    print(full[0].shape)
    full[0] = full[0][:, :, :, :, 0]
    # put it inside a Pytorch form. The image_Dataset only convert Image to Tensor
    full_dataset_train = image_Dataset(image_paths=full[0][0:train_idx], target_paths=full[1][:train_idx])
    full_dataset_val = image_Dataset(image_paths=full[0][train_idx:validation_idx],
                                  target_paths=full[1][train_idx:validation_idx])

    # show the number of image to be processed during training
    print('Training on' + str(len(full_dataset_train)) + 'pictures')
    train_loader = DataLoader(full_dataset_train, batch_size=2,
                              shuffle=False,
                              num_workers=0)
    val_loader = DataLoader(full_dataset_val, batch_size=2,
                            shuffle=False,
                            num_workers=0)
    print('generating model')

    model = m.ModelCountception_v2(inplanes=1, outplanes=1)
    if cuda_available:
        model = model.cuda()
    model = model.double()

    if conf['previous_weights'] != '':
        print('loading previous weights')
        model.load_state_dict(torch.load(conf['previous_weights'])['model_weights'])

    # criterion can be loss_l1 or loss_l2
    criterion = loss_l2
    solver = optim.Adam(model.parameters(), lr=0.00005)
    # if you need  focal dice : loss_fcd = FocalDiceLoss()
    best_val_loss = 10000
    patience = 0
    print('training')
    # For now all is designed for CUDA.
    try:
        for epoch in range(conf['num_epochs']):
            for idx, (inputs, target) in enumerate(train_loader):
                if cuda_available:
                    inputs = inputs.cuda()
                    target = target.cuda()
                output = model.forward(inputs)
                loss = criterion(output, target)
                loss_wing = AdapWingLoss(output, target)
                loss_dice = dice_loss(output, target)

            # Zero grad
                model.zero_grad()
                loss.backward(retain_graph=True)
                loss_wing.backward(retain_graph=True)
                loss_dice.backward()
                solver.step()

            with torch.no_grad():
                print('val_mode')
                val_loss = []
                for idx, (inputs, target) in enumerate(val_loader):
                    if cuda_available:
                        inputs = inputs.cuda()
                        target = target.cuda()
                    output = model.forward(inputs)
                    #every X epochs we save an image that show the ouput heatmap to check improvement
                    if conf['save_heatmap'] !=0 :
                        if (epoch + 1) % conf['save_heatmap'] == 0:
                            heat = output.data.cpu().numpy()
                            plt.imshow(heat[0, 0, :, :] > 0.5)
                            plt.show()
                            plt.savefig('heatmap ' + str(epoch) + '.png')
                    val_loss.append(criterion(output, target).item())

                print("Epoch", epoch, "- Validation Loss:", np.mean(val_loss))

        # best model is saved.
            if abs(np.mean(val_loss)) < best_val_loss and abs(abs(np.mean(val_loss)-best_val_loss)) > 0.5:
                print('New best loss, saving...')
                best_val_loss = copy.deepcopy(np.mean(val_loss))
                patience = 0
                if conf['saved_model'] != '':
                    name = conf['saved_model']
                else:
                    name = 'Countception_train_defaultsave.model'
                state = copy.deepcopy({'model_weights': model.state_dict()})
                torch.save(state, name)
            else:
                patience += 1
            if patience > 10:
                break

    except KeyboardInterrupt:
        print('saving model')
        name = 'checkpoints/Countception_KeyboardInterrupt_save.model'
        state = {'model_weights': model.state_dict()}
        torch.save(state, name)

if __name__ == "__main__":
    main()

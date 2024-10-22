# Author: Lucas
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file license.md

from Data2array import *
from train_utils import *
from models import *
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
from os import path


def main():
    cuda_available = torch.cuda.is_available()
    # import configuration
    conf = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
    # Path to data is stored in config file
    path = conf['path_to_data']
    print('load dataset')
    goal = conf['c2_or_all']
    print(goal)
    ds = load_Data_Bids2Array(path, mode=conf['mode'], split='train', aim=goal)
    print('creating heatmap')
    full = extract_groundtruth_heatmap(ds)

    # put it in the torch loader
    # The 60 first pourcent are for training the 25 next are for validation in an attempt to keep the 15 last for test
    train_idx = int(np.round(len(full[0]) *0.9))
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

    if conf['model'] == 'CC':
        model = ModelCountception_v2(inplanes=1, outplanes=1)
    elif conf['model'] == 'AttU':
        model = AttU_Net()

    if cuda_available:
        model = model.cuda()
    model = model.float()

    if conf['previous_weights'] != '':
        # if path.exist(conf['previous_weights']):
        print('loading previous weights')
        model.load_state_dict(torch.load(conf['previous_weights'])['model_weights'])
    # else:
    #   print('wrong weights path. Starting with random initialization')

    # criterion can be loss_l1 or loss_l2
    criterion = loss_l2
    #model = model.float()

    solver = optim.Adam(model.parameters(), lr=0.0005)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(solver, 32, eta_min=0.00000005, last_epoch=-1)
    # if you need  focal dice : loss_fcd = FocalDiceLoss()
    best_val_loss = 10000
    patience = 0
    print('training')
    # For now all is designed for CUDA.
    try:
        for epoch in range(conf['num_epochs']):
            for idx, (inputs, target) in enumerate(train_loader):
                if cuda_available:
                    inputs = inputs.float().cuda()
                    target = target.float().cuda()
                output = model.forward(inputs)
                loss = criterion(output, target)
                #loss_wing = AdapWingLoss(output, target)
                loss_dice = dice_loss(output, target)

                # Zero grad
                model.zero_grad()
                loss.backward(retain_graph=True)
                #loss_wing.backward(retain_graph=True)
                loss_dice.backward()
                solver.step()
                # print(scheduler.get_lr())
                # scheduler.step()

            with torch.no_grad():
                print('val_mode')
                val_loss = []
                for idx, (inputs, target) in enumerate(val_loader):
                    if cuda_available:
                        inputs = inputs.float().cuda()
                        target = target.float().cuda()
                    output = model.forward(inputs)
                    # every X epochs we save an image that show the ouput heatmap to check improvement
                    if conf['save_heatmap'] != 0:
                        if (epoch + 1) % conf['save_heatmap'] == 0:
                            inp = inputs.data.cpu().numpy()
                            heat = output.data.cpu().numpy()
                            plt.imshow(inp[0, 0, :, :])
                            plt.show()
                            plt.savefig('input' + str(epoch) + '.png')
                            target_i = target.data.cpu().numpy()
                            plt.imshow(target_i[0, 0, :, :])
                            plt.show()
                            plt.savefig('gt' + str(epoch) + '.png')
                            plt.imshow(heat[0, 0, :, :])
                            plt.show()
                            plt.savefig('heatmap ' + str(epoch) + '.png')
                    val_loss.append(criterion(output, target).item())

                print("Epoch", epoch, "- Validation Loss:", np.mean(val_loss))

            # best model is saved.
            if abs(np.mean(val_loss)) < best_val_loss and abs(abs(np.mean(val_loss) - best_val_loss)) > 0.1:
                print('New best loss, saving...')
                best_val_loss = copy.deepcopy(abs(np.mean(val_loss)))
                patience = 0
                if conf['saved_model'] != '':
                    name = conf['saved_model']
                else:
                    name = 'train_defaultsave.model'
                state = copy.deepcopy({'model_weights': model.state_dict()})
                torch.save(state, name)
            else:
                patience += 1
            if patience > conf['patience']:
                break

    except KeyboardInterrupt:
        print('saving model')
        name = 'checkpoints/KeyboardInterrupt_save.model'
        state = {'model_weights': model.state_dict()}
        torch.save(state, name)


if __name__ == "__main__":
    main()

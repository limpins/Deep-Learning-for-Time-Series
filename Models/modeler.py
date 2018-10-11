"""
Email: autuanliu@163.com
Date: 2018/10/10
"""

import shutil
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')


class Modeler:
    """模型训练的一个类
    """

    def __init__(self, network, opt, criterion, dataloaders, lrs_decay, epochs, hidden):
        self.model = network
        self.opt = opt
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.lrs_decay = lrs_decay
        self.epochs = epochs
        self.hidden = hidden

    def train(self):
        """Train and valid(model training and validing each epoch)."""

        loss_t = {'train': [], 'valid': []}
        acc_t = {'train': [], 'valid': []}
        for epoch in range(self.epochs):
            # Each epoch has a training and validation phase
            print(f'Epoch {(epoch + 1):5d}/{self.epochs}')
            for phrase in ['train', 'valid']:
                if phrase == 'train':
                    if 'lrs_decay' is not None:
                        # update learning rates
                        self.lrs_decay.step()
                    self.model.train()
                else:
                    self.model.eval()

                # record the current epoch loss and corrects
                cur_loss = 0.

                # train over minibatch
                for _, (data, target) in enumerate(self.dataloaders[phrase]):
                    self.model = self.model.to(device)
                    data, target = data.to(device), target.to(device)
                    if phrase == 'train':
                        # zero the buffer of parameters' gradient
                        self.opt.zero_grad()
                    # forward
                    out = self.model(data.double, self.hidden)
                    loss = self.criterion(out, target)

                    # backward in training phrase
                    if phrase == 'train':
                        # zero the buffer of parameters' gradient
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                    # statistics
                    cur_loss += loss.item() * data.size(0)
                epoch_loss = cur_loss / len(self.dataloaders[phrase].dataset)
                # save loss and acc
                loss_t[phrase].append(epoch_loss)

                print(f'{phrase}: Loss: {epoch_loss:.4f}')

        # save model
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'best_prec1': self.best_acc,
            'optimizer': self.opt.state_dict(),
        }
        self.save_checkpoint(checkpoint, 'checkpoint.pth.tar')

    def test(self):
        self.model.eval()
        test_loss, correct = 0, 0
        for _, (data, target) in enumerate(self.dataloaders['test']):
            self.model = self.model.to(device)
            data, target = data.to(device), target.to(device)
            output = self.model(data.double, self.hidden)
            # sum up batch loss
            test_loss += self.criterion(output, target).item()
            # get the index of the max
            _, y_pred = torch.max(output.data, 1)
            correct += torch.sum(y_pred == target.data)

        test_loss /= len(self.dataloaders['test'].dataset)
        len1 = len(self.dataloaders['test'].dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len1} ({100. * correct / len1:.0f}%)')

    def predit(self, x):
        pass


    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """Save checkpoint and best model."""
        torch.save(state, filename)
        # shutil.copyfile(filename, self.checkpoint / filename)
        shutil.move(filename, self.checkpoint / filename)

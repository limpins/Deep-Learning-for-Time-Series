"""
Email: autuanliu@163.com
Date: 2018/10/10
Ref: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
"""

import torch
import torch.nn.functional as F
from torch import nn


class Modeler:
    """A base class for model training, validation, prediction etc.

    Args:
            network (nn.Module): instance of defined model without device allocated. 
            opt (torch.optim): the optimizer for network training.
            criterion (nn.Module): the criterion for network training.
            device (torch.device): the device setting for network training.
            batchsize (int, optional): Defaults to 32.
    """

    def __init__(self, network, opt, criterion, device, batchsize=32):
        self.model = nn.DataParallel(network).to(device) if torch.cuda.device_count() > 1 else network.to(device)
        self.opt = opt
        self.criterion = criterion
        self.batchsize = batchsize
        self.dev = device

    def train_model(self, loaders):
        """train model on each epoch.

        Args:
            loaders (DataLoader): dataset for training.

        Returns:
            [float]: elementwise mean loss on each epoch.
        """

        self.model.train()
        hidden = self.model.initHidden(self.batchsize)

        # train over minibatch
        for data, target in loaders:
            data, target = data.to(self.dev).float(), target.to(self.dev).float()
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = self.model.repackage_hidden(hidden)

            # forward
            out, hidden = self.model(data, hidden)
            loss = self.criterion(out, target)

            # backward in training phrase
            # zero the buffer of parameters' gradient
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()

    def evaluate_model(self, loaders):
        """evaluate or test modelself.
        
        Args:
            loaders (DataLoader): dataset for evaluatingself.
        
        Returns:
            float: elementwise mean loss on each epoch.
        """

        self.model.eval()
        hidden = self.model.initHidden(self.batchsize)

        with torch.no_grad():
            # test over minibatch
            for data, target in loaders:
                data, target = data.to(self.dev).float(), target.to(self.dev).float()
                out, hidden = self.model(data, hidden)
                loss = self.criterion(out, target)
        return loss.item()

    def predit_point_by_point(self, x):
        pass

    def save_trained_model(self, path):
        """save trained model's weightsself.
        
        Args:
            path (str): the path to save checkpoint.
        """
        
        # save model weights
        torch.save(self.model.state_dict(), path)

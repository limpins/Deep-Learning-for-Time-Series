"""
Email: autuanliu@163.com
Date: 2018/10/10
Ref: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
"""

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn


class Modeler:
    """A base class for model training, validation, prediction etc.

    Args:
            network (nn.Module): instance of defined model without device allocated.
            opt (torch.optim): the optimizer for network training.
            criterion (nn.Module): the criterion for network training.
            device (torch.device): the device setting for network training.
            visualization (bool, optional): Defaults to False.
    """

    def __init__(self, network, opt, criterion, device, visualization=False):
        self.model = nn.DataParallel(network).to(device) if torch.cuda.device_count() > 1 else network.to(device)
        self.opt = opt
        self.criterion = criterion
        self.dev = device
        self.tsfm = lambda a: a.to(self.dev).float()
        self.visualization = visualization
        if self.visualization:
            self.write = SummaryWriter('log/')

    def __del__(self):
        """close the tensorboard write."""
        if self.visualization:
            self.write.close()

    def train_model(self, loaders, epoch=None):
        """train model on each epoch.

        Args:
            loaders (DataLoader): dataset for training.
            epoch (int): if epoch is an interger, then visualization.

        Returns:
            [float]: elementwise mean loss on each epoch.
        """

        self.model.train()

        # train over minibatch
        for data, target in loaders:
            data, target = self.tsfm(data), self.tsfm(target)

            # forward
            out = self.model(data)
            loss = self.criterion(out, target)

            # backward in training phrase
            # zero the buffer of parameters' gradient
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        if epoch is not None:
            self.write.add_scalar('train loss', loss.item(), epoch)
        return loss.item()

    @torch.no_grad()
    def evaluate_model(self, loaders, epoch=None):
        """evaluate or test modelself.

        Args:
            loaders (DataLoader): dataset for evaluatingself.
            epoch (int): Defaults to None

        Returns:
            float: elementwise mean loss on each epoch.
        """

        self.model.eval()

        # test over minibatch
        for data, target in loaders:
            data, target = self.tsfm(data), self.tsfm(target)
            out = self.model(data)
            loss = self.criterion(out, target)
        if epoch is not None:
            self.write.add_scalar('evaluate loss', loss.item(), epoch)
        return loss.item()

    @torch.no_grad()
    def predit_point_by_point(self, x, y):
        """predict the sequence with point by point method on the whole sequence.

        Args:
            x (torch.tensor): input data.
            y (torch.tensor): target.

        Returns:
            out (torch.tensor): output, prediction, num_point*out_dim.
            err (float): prediction error. (pred - target), num_point*out_dim.
        """

        x, y = self.tsfm(x), self.tsfm(y)
        out = self.model(x).detach()
        return out, out - y

    def save_trained_model(self, path):
        """save trained model's weights.

        Args:
            path (str): the path to save checkpoint.
        """

        # save model weights
        torch.save(self.model.state_dict(), path)

    def save_model(self, path):
        """save model.

        Args:
            path (str): the path to save checkpoint.
        """

        # save model weights
        torch.save(self.model, path)

    def load_best_model(self, path):
        """load best model's weights.

        Args:
            path (str): the path to saved checkpoint.
        """

        # load model weights
        self.model.load_state_dict(torch.load(path))

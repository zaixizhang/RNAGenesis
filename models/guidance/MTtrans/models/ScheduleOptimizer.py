import torch
import os
import sys
import numpy as np 
import torch
from torch import nn
from matplotlib import pyplot as plt
from matplotlib import cm

global scheduleoptim_text
scheduleoptim_text="ScheduledOptim(optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),n_warmup_steps=20)"

scheduleoptim_dict_str="""ScheduledOptim(optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                                    betas=(0.9, 0.98),
                                                    eps=1e-09, 
                                                    weight_decay={weight_decay}, 
                                                    amsgrad={amsgrad}),
                                         n_warmup_steps={n_warmup_steps})"""

class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        return self.optimizer.state_dict()
        
    def load_state_dict(self,state):
        self.optimizer.load_state_dict(state)

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2
        self.delta = min(1024,self.delta)

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
            # -1.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def find_lr(net,train_data,Variable):
    criterion = torch.nn.CrossEntropyLoss()
    
    net.fc = nn.Linear(2048, 120)

    with torch.cuda.device(0):
        net = net.cuda()

    basic_optim = torch.optim.SGD(net.parameters(), lr=1e-5)
    optimizer = ScheduledOptim(basic_optim)


    lr_mult = (1 / 1e-5) ** (1 / 100)
    lr = []
    losses = []
    best_loss = 1e9
    for data, label in train_data:
        with torch.cuda.device(0):
            data = Variable(data.cuda())
            label = Variable(label.cuda())
        # forward
        out = net(data)
        loss = criterion(out, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr.append(optimizer.learning_rate)
        losses.append(loss.data[0])
        optimizer.set_learning_rate(optimizer.learning_rate * lr_mult)
        if loss.data[0] < best_loss:
            best_loss = loss.data[0]
        if loss.data[0] > 4 * best_loss or optimizer.learning_rate > 1.:
            break

    plt.figure()
    plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.plot(np.log(lr), losses)
    plt.show()
    plt.figure()
    plt.xlabel('num iterations')
    plt.ylabel('learning rate')
    plt.plot(lr)
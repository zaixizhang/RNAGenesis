import os
import torch
from torch import nn
import numpy as np


def softmax(x):
    """
    one dimensional softmax designed for numpy array
    """
    e_x = np.exp(x)
    out = e_x / e_x.sum()
    return out

class Dynamic_Weight_Averaging():
    def __init__(self,tasks,tau,init_weight):
        """
        Dynamic Weight Averaging implementation
        """
        # number of tasks
        self.N = len(tasks)
        # self.loss_list = np.array([loss_dict[t+'_loss'].detach().cpu().item() for t in tasks])
        self.step = 0
        self.omega = np.array([init_weight]*self.N ) # the weight , \omega_i(t)
        self.tau = tau # † : a value to adjust the soft-max
    
    def _magnitude_adjust(self):
        """
        automatic adjustment of loss magnitude
        """
        self.relative_magnitude = self.loss_list.min() / self.loss_list
    
    def  _update(self,loss_dict):
        # update r with L_i(t-1) 
        self.step += 1
        if self.step < 2:
            self.loss_list = np.array([loss_dict[t+'_loss'].detach().cpu().item() for t in self.tasks])
            return self.init_weight
        else:
            self._magnitude_adjust()
            last_loss = self.loss_list
            self.loss_list = np.array([loss_dict[t+'_loss'].detach().cpu().item() for t in self.tasks])
            # computing DWA
            r_t = np.divide(self.loss_list,last_loss) / self.tau
            self.omega = self.N * softmax(r_t)
            weight_t = np.multiply(self.relative_magnitude,self.omega)
            return {self.tasks[i]:weight_t[i] for i in range(len(self.tasks))}
    
class Dynamic_Task_Priority(object):
    def __init__(self,tasks,gamma,init_weight):
        """
        Dynamic Task Priority (DTP) wish to weight more on tasks with lower KPI
        In our cases, we take accuracy as the KPI
        """
        self.gamma = np.array([gamma[t] for t in tasks]) if type(gamma) == dict else gamma # an adjustment param
        self.tasks = tasks
        # self.kappa = [loss_dict[t+'_Acc'] for t in tasks]
        self.omega = init_weight 
        self.init_weight = init_weight
        self.step = 0
            
    
    def _magnitude_adjust(self):
        """
        automatic adjustment of loss magnitude
        """
        # TODO : test this effectiveness
        self.relative_magnitude = self.kappa.min() / (self.kappa+1e-8)
        if np.any(self.kappa==1):
            self.kappa[np.where(self.kappa==1)[0]] -= 1e-8
    
    def _update(self,loss_dict):
        """
        weight is updated like belowing
        `math` : w_i(t) = -(1-\kappa_i(t))^{\gamma_i} log \kappa_i(t)
        """
        self.step += 1
        self.kappa = np.array([loss_dict[t+'_Acc'] for t in self.tasks])
        self._magnitude_adjust()
        self.omega = -1 * np.multiply(np.power(1 - self.kappa,self.gamma),np.log(self.kappa+1e-8))
        
        if self.step < 2:
            return self.init_weight
        else:
            weight_t = np.multiply(self.relative_magnitude,self.omega)
            return {self.tasks[i]:weight_t[i] for i in range(len(self.tasks))}
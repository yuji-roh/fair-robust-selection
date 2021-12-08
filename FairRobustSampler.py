import sys, os
import numpy as np
import math
import random
import itertools
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch


class CustomDataset(Dataset):
    """Custom Dataset.

    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (true labels) of data.
        z: A PyTorch tensor for z features (sensitive attributes) of data.
    """
    def __init__(self, x_tensor, y_tensor, z_tensor):
        """Initializes the dataset with torch tensors."""
        
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor
        
    def __getitem__(self, index):
        """Returns the selected data based on the index information."""
        
        return (self.x[index], self.y[index], self.z[index])

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.x)
    
    
class FairRobust(Sampler):
    """FairRobust (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch [Roh et al., ICLR 2021] with robust training.

    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type 
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the indexes of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        clean_index: A list that contains the data indexes of selected samples.
        clean_y_, clean_z_, clean_yz_index: Dictionaries containing the indexes of each class in the selected set.
        clean_y_, clean_z_, clean_yz_len: Dictionaries containing the length information in the selected set.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values for fairness [Roh et al., ICLR 2021].
        tau: (0~1) real number indicating the clean ratio of the data.
        warm_start: An integer for warm-start period.

        
    """
    def __init__(self, model, x_tensor, y_tensor, z_tensor, target_fairness, parameters, replacement = False, seed = 0):
        """Initializes FairBatch."""
        
        self.model = model
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.x_data = x_tensor
        self.y_data = y_tensor
        self.z_data = z_tensor
        
        
        self.alpha = parameters.alpha
        self.fairness_type = target_fairness
        
        self.replacement = replacement
        
        self.N = len(z_tensor)
        
        self.batch_size = parameters.batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)
        
        # Takes the unique values of the tensors
        self.z_item = list(set(z_tensor.tolist()))
        self.y_item = list(set(y_tensor.tolist()))
        
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))
        
        # Makes masks
        self.z_mask = {}
        self.y_mask = {}
        self.yz_mask = {}
        
        for tmp_z in self.z_item:
            self.z_mask[tmp_z] = (self.z_data == tmp_z)
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yz in self.yz_tuple:
            self.yz_mask[tmp_yz] = (self.y_data == tmp_yz[0]) & (self.z_data == tmp_yz[1])
        

        # Finds the index
        self.z_index = {}
        self.y_index = {}
        self.yz_index = {}
        
        for tmp_z in self.z_item:
            self.z_index[tmp_z] = (self.z_mask[tmp_z] == 1).nonzero().squeeze()
            
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()
        
        for tmp_yz in self.yz_tuple:
            self.yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1).nonzero().squeeze()
            
        self.entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])
            
        # Length information
        self.z_len = {}
        self.y_len = {}
        self.yz_len = {}
        
        for tmp_z in self.z_item:
            self.z_len[tmp_z] = len(self.z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            self.yz_len[tmp_yz] = len(self.yz_index[tmp_yz])

        # Default batch size
        self.S = {}
        
        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.yz_len[tmp_yz])/self.N

        
        self.lb1 = (self.S[1,1])/(self.S[1,1]+(self.S[1,0]))
        self.lb2 = (self.S[-1,1])/(self.S[-1,1]+(self.S[-1,0]))
        
        # For cleanselection parameters
        self.tau = parameters.tau # Clean ratio
        self.warm_start = parameters.warm_start
    
        self.count_epoch = 0
        
            
        # Clean sample selection
        self.clean_index = np.arange(0,len(self.y_data))
        
        # Finds the index
        self.clean_z_index = {}
        self.clean_y_index = {}
        self.clean_yz_index = {}
        
        for tmp_z in self.z_item:
            self.clean_z_index[tmp_z] = (self.z_mask[tmp_z] == 1)[self.clean_index].nonzero().squeeze()
            
        for tmp_y in self.y_item:
            self.clean_y_index[tmp_y] = (self.y_mask[tmp_y] == 1)[self.clean_index].nonzero().squeeze()
        
        for tmp_yz in self.yz_tuple:
            self.clean_yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1)[self.clean_index].nonzero().squeeze()
        
        
       # Length information
        self.clean_z_len = {}
        self.clean_y_len = {}
        self.clean_yz_len = {}
        
        for tmp_z in self.z_item:
            self.clean_z_len[tmp_z] = len(self.clean_z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.clean_y_len[tmp_y] = len(self.clean_y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            self.clean_yz_len[tmp_yz] = len(self.clean_yz_index[tmp_yz])
 
      
    def get_logit(self):
        """Runs forward pass of the intermediate model with the training data.
        
        Returns:
            Outputs (logits) of the model.

        """
        
        self.model.eval()
        logit = self.model(self.x_data)
        
        return logit
    
    
    def adjust_lambda(self, logit):
        """Adjusts the lambda values using FairBatch [Roh et al., ICLR 2021].
        See our paper for algorithm details.
        
        Args: 
            logit: A torch tensor that contains the intermediate model's output on the training data.
        
        """
        
        criterion = torch.nn.BCELoss(reduction = 'none')
        
        
        if self.fairness_type == 'eqopp':
            
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.clean_yz_index[tmp_yz]])) / self.clean_yz_len[tmp_yz]
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / self.clean_y_len[tmp_y]
            
            # lb1 * loss_z1 + (1-lb1) * loss_z0
            
            if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                self.lb1 += self.alpha
            else:
                self.lb1 -= self.alpha
                
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1 
                
        elif self.fairness_type == 'eqodds':
            
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.clean_yz_index[tmp_yz]])) / (self.clean_yz_len[tmp_yz]+1)
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / (self.clean_y_len[tmp_y]+1)
            
            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]:
                    self.lb2 += self.alpha
                else:
                    self.lb2 -= self.alpha
                    
                
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1
                
            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1
                
        elif self.fairness_type == 'dp':
            yhat_yz = {}
            yhat_y = {}
            
            ones_array = np.ones(len(self.y_data))
            ones_tensor = torch.FloatTensor(ones_array).cuda()
            dp_loss = criterion((F.tanh(logit.squeeze())+1)/2, ones_tensor.squeeze()) # Note that ones tensor puts as the true label
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(dp_loss[self.clean_yz_index[tmp_yz]])) / self.clean_z_len[tmp_yz[1]]
                    
            
            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]: 
                    self.lb2 -= self.alpha
                else:
                    self.lb2 += self.alpha
                    
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1
                
            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1

    def select_fair_robust_sample(self):
        """Selects fair and robust samples and adjusts the lambda values for fairness. 
        See our paper for algorithm details.
        
        Returns:
            Indexes that indicate the data.
            
        """
        
        logit = self.get_logit()
        
        self.adjust_lambda(logit)

        criterion = torch.nn.BCELoss(reduction = 'none')
        
        loss = criterion ((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
        profit = torch.max(loss)-loss
        
        current_weight_sum = {}
        
        lb_ratio = {}
        
        for tmp_yz in self.yz_tuple:
            if tmp_yz == (1,1):
                lb_ratio[tmp_yz] = self.lb1
            elif tmp_yz == (1,0): 
                lb_ratio[tmp_yz] = 1-self.lb1
            elif tmp_yz == (-1,1):
                lb_ratio[tmp_yz] = self.lb2
            elif tmp_yz == (-1,0):
                lb_ratio[tmp_yz] = 1-self.lb2
            
            current_weight_sum[tmp_yz] = 0
        
        # Greedy-based algorithm
        
        (_, sorted_index) = torch.topk(profit, len(profit), largest=True, sorted=True)
        
        clean_index = []
        
        total_selected = 0
        
        desired_size = int(self.tau * len(self.y_data))
        
        for j in sorted_index:
            tmp_y = self.y_data[j].item()
            tmp_z = self.z_data[j].item()
            current_weight_list = list(current_weight_sum.values())
            
            if total_selected >= desired_size:
                break
            if all(i < desired_size for i in current_weight_list):
                clean_index.append(j)
                
                current_weight_sum[(tmp_y, tmp_z)] += 2 - lb_ratio[(tmp_y, tmp_z)]
                current_weight_sum[(tmp_y, 1-tmp_z)] += 1 - lb_ratio[(tmp_y, 1-tmp_z)]
                current_weight_sum[(tmp_y * -1, tmp_z)] += 1
                current_weight_sum[(tmp_y * -1, 1-tmp_z)] += 1
                
                total_selected += 1        
        
        clean_index = torch.LongTensor(clean_index).cuda()
        
        self.batch_num = int(len(clean_index)/self.batch_size)
        
        # Update the variables
        self.clean_index = clean_index
        
        for tmp_z in self.z_item:
            combined = torch.cat((self.z_index[tmp_z], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_z_index[tmp_z] = intersection 
            
        for tmp_y in self.y_item:
            combined = torch.cat((self.y_index[tmp_y], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_y_index[tmp_y] = intersection
        
        for tmp_yz in self.yz_tuple:
            combined = torch.cat((self.yz_index[tmp_yz], self.clean_index))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            
            self.clean_yz_index[tmp_yz] = intersection
        
        
        for tmp_z in self.z_item:
            self.clean_z_len[tmp_z] = len(self.clean_z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.clean_y_len[tmp_y] = len(self.clean_y_index[tmp_y])
        
        for tmp_yz in self.yz_tuple:
            self.clean_yz_len[tmp_yz] = len(self.clean_yz_index[tmp_yz])
            
        
        return clean_index
        
    
    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False, weight = None):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indexes that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                if weight == None:
                    weight_norm = weight/torch.sum(weight)
                    select_index.append(np.random.choice(full_index, batch_size, replace = False, p = weight_norm))
                else:
                    select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    
                    start_idx = len(full_index)-start_idx
                else:

                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
        return select_index


    
    def decide_fair_batch_size(self):
        """Calculates each class size based on the lambda values (lb1 and lb2) for fairness.
        
        Returns:
            Each class size for fairness.
            
        """
        
        each_size = {}

        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.clean_yz_len[tmp_yz])/len(self.clean_index)

        # Based on the updated lambdas, determine the size of each class in a batch
        if self.fairness_type == 'eqopp':
            # lb1 * loss_z1 + (1-lb1) * loss_z0

            each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(-1,1)] = round(self.S[(-1,1)])
            each_size[(-1,0)] = round(self.S[(-1,0)])

        elif self.fairness_type == 'eqodds':
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(-1,1)] = round(self.lb2 * (self.S[(-1,1)] + self.S[(-1,0)]))
            each_size[(-1,0)] = round((1-self.lb2) * (self.S[(-1,1)] + self.S[(-1,0)]))

        elif self.fairness_type == 'dp':
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(-1,1)] = round(self.lb2 * (self.S[(-1,1)] + self.S[(-1,0)]))
            each_size[(-1,0)] = round((1-self.lb2) * (self.S[(-1,1)] + self.S[(-1,0)]))
        
        return each_size
        
        
    
    def __iter__(self):
        """Iters the full process of fair and robust sample selection for serving the batches to training.
        
        Returns:
            Indexes that indicate the data in each batch.
            
        """
        self.count_epoch += 1
        
        if self.count_epoch > self.warm_start:

            _ = self.select_fair_robust_sample()


            each_size = self.decide_fair_batch_size()

            # Get the indices for each class
            sort_index_y_1_z_1 = self.select_batch_replacement(each_size[(1, 1)], self.clean_yz_index[(1,1)], self.batch_num, self.replacement)
            sort_index_y_0_z_1 = self.select_batch_replacement(each_size[(-1, 1)], self.clean_yz_index[(-1,1)], self.batch_num, self.replacement)
            sort_index_y_1_z_0 = self.select_batch_replacement(each_size[(1, 0)], self.clean_yz_index[(1,0)], self.batch_num, self.replacement)
            sort_index_y_0_z_0 = self.select_batch_replacement(each_size[(-1, 0)], self.clean_yz_index[(-1,0)], self.batch_num, self.replacement)

            for i in range(self.batch_num):
                key_in_fairbatch = sort_index_y_0_z_0[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_0[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_0_z_1[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_1[i].copy()))

                random.shuffle(key_in_fairbatch)

                yield key_in_fairbatch

        else:
            entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])

            sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)

            for i in range(self.batch_num):
                yield sort_index[i]
        
                               

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.y_data)


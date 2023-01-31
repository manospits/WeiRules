import frcnn as fw
import pandas as pd
import numpy as np
import glob
import os


import torch
from PIL import Image
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import torch.nn.functional as F
from collections import Counter


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.distributed as dist


import math
import sys
import time


import matplotlib.pyplot as plt
import locale

from torch import autograd

from torch.utils import data
import torchvision


###########################################################################################
############### YahooDataset ##############################################################
###########################################################################################

def create_boxes_and_labels(box,imwidth,imheight,label):
    xmin, ymin, xmax, ymax = box
    boxes=[[float(xmin), float(ymin), float(xmax), float(ymax)]]
    labels=[label]
    return boxes,labels

class YahooDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y, rule_input_start, imgpath, dfdict=None):
        'Initialization'
        self.X = X
        self.Y = Y
        self.imgpath = imgpath
        self.rule_input_start=rule_input_start
        self.dfdict = dfdict
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        row_x = self.X[index]
        row_y = self.Y[index]
        x_id = row_x[0]
        image_path = row_x[1]
        rule_input = torch.tensor([row_x[self.rule_input_start:].astype(float)])
        if self.dfdict==None:
            or_im = np.array(Image.open(self.imgpath+image_path).convert("RGB"))
            im_size=np.array(or_im).shape
            imheight=im_size[1]
            imwidth=im_size[2]
            box=row_x[2:6]
            compose = transforms.Compose([transforms.ToTensor()])
            deep_input=compose(np.array(or_im))
            boxes, labels = create_boxes_and_labels(box,imwidth,imheight, row_y)
            return torch.tensor(x_id), deep_input, rule_input, torch.tensor(boxes), torch.tensor(labels,dtype=torch.int64)
        else:
            deep_feature = self.dfdict[x_id][0]
            label=row_y
            return torch.tensor(x_id), rule_input, torch.tensor(deep_feature), torch.tensor(label)


###########################################################################################
############### WeiRules ##################################################################
###########################################################################################

def predict_weirules(weirules_model, test_generator):
    weirules_model.model.eval()
    all_results=[]
    for local_image_num, local_rule_input, local_deep_features, local_labels in test_generator:      
            
            
            local_deep_features = torch.stack([x.float().cuda() for x in local_deep_features],dim=0)
            local_rule_input = torch.stack([x[0].float().cuda() for x in local_rule_input],dim=0)
            local_labels = torch.stack([x.cuda() for x in local_labels],dim =0)


            wei_logits = weirules_model.forward_model(local_deep_features, local_rule_input)
            probs=F.normalize(wei_logits, p=1 ,dim=1) 
            
            local_labels=local_labels.cpu().detach().numpy()
            batch_predictions=torch.argmax(probs,dim=1).cpu().detach().numpy()
            for pred, label in zip(batch_predictions, local_labels):
                all_results.append((pred,label))
            del local_image_num
            del local_deep_features
            del local_rule_input
    return all_results
            
            

            
            
def train_weirules(weirules_model, train_generator, optimizer, max_epochs, val_generator=None):
    wrunning_losses=[]
    for epoch in range(max_epochs):
        n_batches=0
        wrunning_loss=0
        weirules_model.model.train()

        for _, local_rule_input, local_deep_features, local_labels in train_generator:      
            local_deep_features = torch.stack([x.float().cuda() for x in local_deep_features],dim=0)
            local_rule_input = torch.stack([x[0].float().cuda() for x in local_rule_input],dim=0)
            local_labels = torch.stack([x.cuda() for x in local_labels],dim =0)
            wei_logits = weirules_model.forward_model(local_deep_features, local_rule_input)
            wei_loss = weirules_model.model.cross_entropy(wei_logits, local_labels)
            #model weirules
            wloss_value = wei_loss.item()
            wrunning_loss+=wloss_value
            optimizer.zero_grad()
            with autograd.detect_anomaly():
                wei_loss.backward()
            optimizer.step()
            
            del wei_loss
            del local_deep_features
            del local_rule_input
            del local_labels
            n_batches+=1
            print('.', end='')
        else:
            print('')
            all_best_results=predict_weirules(weirules_model, train_generator)
            wlabel = []
            alabel = []
            for res in all_best_results:
                wlabel.append(res[0])
                alabel.append(res[1])

            wei_train_f = classification_report(alabel, wlabel,output_dict=True)['macro avg']['f1-score']
            print(f"\n+---Epoch: {epoch} \n\t---weirules loss: {wrunning_loss/n_batches}\n\t-------------------------------------------\n\t---weirules train f1 score:{wei_train_f}\n")
            if not val_generator == None:
                all_best_results=predict_weirules(weirules_model, val_generator)
                wlabel = []
                alabel = []
                for res in all_best_results:
                    wlabel.append(res[0])
                    alabel.append(res[1])
                wei_val_f = classification_report(alabel, wlabel,output_dict=True)['macro avg']['f1-score']
                print(f"\t--------------------------------------------\n\t---weirules val f1 score:{wei_val_f}\n")
            wrunning_losses.append(wrunning_loss/n_batches)
    return wrunning_losses



###########################################################################################
############### FasterRCNN ################################################################
###########################################################################################

def train_frcnn(model, train_generator, optimizer, max_epochs, val_generator=None):
    for epoch in range(max_epochs):
        n_batches=0
        frunning_loss=0
        wrunning_loss=0

        model.train()
        frunning_losses=[]
        for _,local_deep_input, local_rule_input, local_boxes, local_labels in train_generator:
            local_deep_input = [x.float().cuda() for x in local_deep_input]
            local_rule_input = torch.stack([x.float().cuda() for x in local_rule_input],dim=0)
            local_targets = []
            for i in range(len(local_deep_input)):
                d = {}
                d['boxes'] = local_boxes[i].cuda()
                d['labels'] = (local_labels[i]+1).cuda()
                local_targets.append(d)
            #model frcnn
            loss_dict = model.forward(local_deep_input, local_targets, None)
            wei_loss = loss_dict.pop('loss_classifier_w')
            f_losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            floss_value = losses_reduced.item()
            frunning_loss+=floss_value

            optimizer.zero_grad()
            with autograd.detect_anomaly():
                f_losses.backward()
            optimizer.step()
            del f_losses
            del wei_loss
            n_batches+=1
            print('.', end='')
        else:
            print('')

            all_best_results=predict_frcnn(model, train_generator)
            flabel = []
            alabel = []
            frcnn_train_f=0
            if len(all_best_results[0])>0:
                for res in all_best_results[0]:
                    flabel.append(res[1]['label']-1)
                    alabel.append(res[2])

            frcnn_train_f = classification_report(alabel, flabel,output_dict=True)['macro avg']['f1-score']
            print(f"\n+---Epoch: {epoch} \n\
            ---frcnn loss: {frunning_loss/n_batches}\n\
            -------------------------------------------\n\
            ---frcnn train f1 score:{frcnn_train_f}\n")
            if not val_generator==None:
                all_best_results=predict_frcnn(model, val_generator)
                flabel = []
                alabel = []
                for res in all_best_results:
                    flabel.append(res[1]['label']-1)
                    alabel.append(res[2])

                frcnn_val_f = classification_report(alabel, flabel,output_dict=True)['macro avg']['f1-score']
                print(f"--------------------------------------------\n\
                ---frcnn val f1 score:{frcnn_val_f}\n")
            frunning_losses.append(frunning_loss/n_batches)


        
def predict_frcnn(model, generator):
    model.eval()
        
    all_results=[]
    all_best_results=[]
    
    ground_truth = []
    predictions = []
    classes=[]
    for local_image_nums, local_deep_input, local_rule_input, local_boxes, local_labels in generator:
        local_deep_input = [x.float().cuda() for x in local_deep_input]
        local_rule_input = torch.stack([x.float().cuda() for x in local_rule_input],dim=0)

        results = model.forward(local_deep_input, targets=None, rule_input=None)
        labels = local_labels
        image_nums = [x.cpu().detach().numpy().tolist() for x in local_image_nums]
        for i, result in enumerate(results):
            no_prediction=False
            label = labels[i][0].cpu().detach().item()
            ground_truth.append([
                image_nums[i],
                label,
                1,
                local_boxes[i][0].cpu().detach().numpy().tolist()
            ])
            if label not in classes:
                classes.append(label)
            for key in result:
                values=[]
                for value in result[key]:
                    values.append(value.cpu().detach().numpy())
                if values==[]:
                    result[key]=[]
                    no_prediction=True
                else:
                    result[key]=np.stack(values,axis=0)  
            if no_prediction:
                continue
            for res_i in range(len(result['scores'])):
                label=result['labels'][res_i]
                if label not in classes:
                    classes.append(int(label))
                predictions.append([image_nums[i],
                                    result['labels'][res_i],
                                    result['scores'][res_i],
                                    result['boxes'][res_i].tolist()
                                   ])
            best_res={}
            findex=np.argmax(result['scores'])
            best_res['fscore']=np.max(result['scores'])
            best_res['box']=result['boxes'][findex]
            best_res['label']=result['labels'][findex]

            all_results.append((image_nums[i],results,labels[i][0].cpu().detach().item()))
            all_best_results.append((image_nums[i], best_res,labels[i][0].cpu().detach().item()))

        del results
        del local_image_nums
        del local_deep_input
        del local_rule_input
        del local_boxes
        del local_labels
        
    return all_best_results, ground_truth, predictions, classes

def extract_df(model, generator):
    model.eval()
        
    all_df={}

    for local_image_nums, local_deep_input, _, local_boxes, local_labels in generator:
        local_deep_input = [x.float().cuda() for x in local_deep_input]
        extracted_df = model.forward(local_deep_input, targets=None, rule_input=None, extract_df=True)
        
        extracted_df=[df.cpu().detach().numpy() for df in extracted_df]
        for i,image_idt in enumerate(local_image_nums):
            image_id = image_idt.cpu().detach().item()
            image_df = extracted_df[i]
            all_df[image_id]=image_df

        del local_image_nums
        del local_deep_input
        del local_boxes
        del local_labels
        
    return all_df

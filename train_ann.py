# coding: utf-8
import os
import time
import sys
import yaml
import numpy as np
import pandas as pd
from src.util import ExeDataset,write_pred
from src.model import MalConv
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import WeightedRandomSampler
import argparse
import torch.nn.functional as F

seed = 11123

np.random.seed(seed)
torch.manual_seed(seed)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_csv', type=str)
args = parser.parse_args()

train_df = pd.read_excel(args.train_csv)
train_df['kvalue'] = train_df['kvalue'].str.replace("math.pi", "3.14")
train_df['kvalue'] = train_df['kvalue'].str.replace("math.tau", "6.28")
train_df['kvalue'] = train_df['kvalue'].str.replace("math.e", "2.72")
train_df['kvalue'] = train_df['kvalue'].str.replace("math.inf", "999999999999")
train_df['kvalue'] = train_df['kvalue'].str.replace("math.nan", "-999999999999")
train_df[['k1', 'k2', 'k3', 'k4']] = train_df['kvalue'].str.split(',',expand=True).astype(float)
train_df = train_df.drop('kvalue', axis=1)
train_df = train_df.sample(frac=1).reset_index(drop=True)
y = train_df['state']
train_df = train_df.drop(['state', 'files'], axis=1)

print("train_df", train_df.head(), list(train_df))


train_df_dummy = pd.get_dummies(train_df)
print("train_dummy_df", train_df_dummy.head(), list(train_df_dummy))

log_file_path = './ann.log'
chkpt_acc_path = './ann.model'
pred_path = './ann.pred'

# Parameters
use_gpu = True
use_cpu = False
learning_rate = 0.00001
max_step = 10000
batch_size = 32
total = len(y)
valid = int(total * 0.8)
# test = int(total * 0.8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_x = torch.from_numpy(train_df_dummy[:valid].values).to(device).float()
valid_x = torch.from_numpy(train_df_dummy[valid:].values).to(device).float()
# test_x = torch.from_numpy(train_df_dummy[test:].values).to(device).float()

train_y = torch.from_numpy(y[:valid].values).to(device).float()
valid_y = torch.from_numpy(y[valid:].values).to(device).float()
# test_y = torch.from_numpy(y[test:].values).to(device).float()


train_set = TensorDataset(train_x, train_y)
dataloader = DataLoader(train_set, batch_size=batch_size)
valid_set = TensorDataset(valid_x, valid_y)
validloader = DataLoader(valid_set,
                        batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(22, 100)  # 5*5 from image dimension
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x.view(-1)


malconv = Net()
bce_loss = nn.BCELoss()
adam_optim = optim.Adam(malconv.parameters(), lr=learning_rate)

if use_gpu:
    malconv = malconv.cuda()
    bce_loss = bce_loss.cuda()


step_msg = 'step-{}-loss-{:.6f}-acc-{:.4f}-time-{:.2f}'
valid_msg = 'step-{}-tr_loss-{:.6f}-tr_acc-{:.4f}-val_loss-{:.6f}-val_acc-{:.4f}'
log_msg = '{}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}'
history = {}
history['tr_loss'] = []
history['tr_acc'] = []

log = open(log_file_path,'w')
log.write('step,tr_loss, tr_acc, val_loss, val_acc, time\n')

valid_best_acc = 0.0
valid_best_precision = 0.0
valid_best_recall = 0.0
valid_best_f1 = 0.0
valid_best_fpr = 0.0
valid_best_fnr = 0.0



total_step = 0
test_step = 10

PATIENCE = 300

local_patience = PATIENCE
idx = 0
while total_step < max_step:
    idx = idx + 1
    # Training 
    for x, label in dataloader:
        adam_optim.zero_grad()
        cur_batch_size = y.size
        pred = malconv(x)
        loss = bce_loss(pred,label)
        loss.backward()
        adam_optim.step()
        history['tr_loss'].append(loss.cpu().data.numpy())
        # history['tr_loss'].append(loss.cpu().data.numpy()[0])
        history['tr_acc'].extend(list(label.cpu().data.numpy().astype(int)==(pred.cpu().data.numpy()+0.5).astype(int)))
    
    # Testing
    history['val_loss'] = []
    history['val_acc'] = []
    history['val_pred'] = []

    # other evaluation metrics..
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for x, label in validloader:
        cur_batch_size = label.size
        pred = malconv(x)
        loss = bce_loss(pred,label)
        
        y_trues = list(label.cpu().data.numpy().astype(int))
        y_preds = list((pred.cpu().data.numpy()+0.5).astype(int))

        for y_true, y_pred in zip(y_trues, y_preds):
            if y_true == 1:
                if y_true == y_pred:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if y_true == y_pred:
                    tn = tn + 1
                else:
                    fp = fp + 1

        history['val_loss'].append(loss.cpu().data.numpy())
        # history['val_loss'].append(loss.cpu().data.numpy()[0])
        history['val_acc'].extend(list(label.cpu().data.numpy().astype(int)==(pred.cpu().data.numpy()+0.5).astype(int)))
        history['val_pred'].append(list(pred.cpu().data.numpy()))

    print(valid_msg.format(idx,np.mean(history['tr_loss']),np.mean(history['tr_acc']),
                           np.mean(history['val_loss']),np.mean(history['val_acc'])))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)    # true positive rate
    f1 = 2*precision*recall / (precision + recall + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    fnr = fn / (fn + tp + 1e-6)

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, FPR: {fpr:.2f}, FNR: {fnr:.2f}")

    
    if (valid_best_f1 < f1):
        local_patience = PATIENCE

        valid_best_acc = np.mean(history['val_acc'])
        valid_best_precision = precision
        valid_best_recall = recall
        valid_best_fnr = fnr
        valid_best_fpr = fpr
        valid_best_f1 = f1

        # torch.save(malconv,chkpt_acc_path)
        # print('Checkpoint saved at',chkpt_acc_path)
        # # write_pred(history['val_pred'],0,pred_path)
        # print('Prediction saved at', pred_path)
    else:
        local_patience = local_patience - 1
        if local_patience < 0:
          break

    history['tr_loss'] = []
    history['tr_acc'] = []

print("==============================END OF TRAINING======================================")
print(f"Accuracy: {valid_best_acc:.2f}, Precision: {valid_best_precision:.2f}, Recall: {valid_best_recall:.2f}, F1: {valid_best_f1:.2f}, FPR: {valid_best_fpr:.2f}, FNR: {valid_best_fnr:.2f}")


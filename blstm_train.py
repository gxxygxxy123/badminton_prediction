from email.policy import default
import torch.nn as nn
import torch
import pandas as pd
import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from datetime import datetime

from blstm import Blstm
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from dataloader import RNNDataSet, PhysicsDataSet_blstm
from predict import predict3d, predict2d

sns.set()

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, device):
    
    model.train()
    
    epoch_loss = 0

    for batch_idx, (data, label) in enumerate(iterator):
        src = data.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, N, IN_SIZE]
        trg = label.float().type(torch.FloatTensor).to(device)

        optimizer.zero_grad()

        output = model(src)  # predict next step, init hidden state to zero at the begining of the sequence
        #output = [BATCH_SIZE, N, OUT_SIZE]

        # or implement other loss function
        loss = criterion(output, trg) # predict next step for each step

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def loss_function(output, trg):
    # output=[batch size, trg_len, 3]
    # trg   =[batch size, trg_len, 3]

    loss = torch.linalg.norm(trg[:,:,:-1]-output[:,:,:-1], axis=2) # x y (m)

    loss += torch.linalg.norm(trg[:,:,-1:]-output[:,:,-1:], axis=2) * 10 # timestamp (s)

    #loss = torch.linalg.norm(pred_points-true_points, axis=1)
    return torch.mean(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BLSTM Training Program")
    parser.add_argument("-s","--seq", type=int, help="Input Sequence", required=True)
    parser.add_argument("-e","--epoch", type=int, help="Training Epochs", required=True)
    parser.add_argument("--hidden_size", type=int, help="Hidden Size", default=32)
    parser.add_argument("--hidden_layer", type=int, help="Hidden Layer", default=1)
    parser.add_argument("--physics_data", type=int, help="Training Datas", default=140000)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=8)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.01) # SGD default lr=0.01
    parser.add_argument("-w","--weight", type=str, help="Ouput Weight name")
    parser.add_argument("--save_epoch", type=int, help="Save at each N epoch", default=200)
    parser.add_argument('--early_stop', action="store_true", help = 'Early Stop')
    args = parser.parse_args()

    N = args.seq # Time Sequence Number
    N_EPOCHS = args.epoch
    HIDDEN_SIZE = args.hidden_size
    N_LAYERS = args.hidden_layer
    N_PHYSICS_DATA = args.physics_data
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    SAVE_EPOCH = args.save_epoch

    print(f"Input Seq: {N}\n"
          f"Epoch: {N_EPOCHS}\n"
          f"Hidden Size: {HIDDEN_SIZE}\n"
          f"Hidden Layer: {N_LAYERS}\n"
          f"Physics data: {N_PHYSICS_DATA}\n"
          f"Batch Size: {BATCH_SIZE}\n"
          f"Learning Rate: {LEARNING_RATE}\n"
          f"Save At Each {SAVE_EPOCH} epoch\n")

    IN_SIZE = 3
    OUT_SIZE = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = PhysicsDataSet_blstm(N=N, datas=N_PHYSICS_DATA)

    train_dataset_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)

    model = Blstm(in_size=IN_SIZE, out_size=OUT_SIZE, hidden_size=HIDDEN_SIZE, hidden_layer=N_LAYERS, device=device)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.to(device)
    model.apply(init_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # criterion = nn.MSELoss(reduction='mean')
    criterion = loss_function
    
    history_train_loss = []

    print(f"Start Training ... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for epoch in range(1,N_EPOCHS+1):

        train_loss = train(model, train_dataset_dataloader, optimizer, criterion, device=device)
        history_train_loss.append(train_loss)

        if epoch % SAVE_EPOCH == 0:
            print(f"Epoch: {epoch}/{N_EPOCHS}. Loss: {train_loss:.4f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            torch.save(model.state_dict(), f'./weight/blstm_weight_s{N}_e{epoch}_p{N_PHYSICS_DATA}')
            print(f"Save weight ./weight/blstm_weight_s{N}_e{epoch}_p{N_PHYSICS_DATA}")

        # Early Stopping if loss > avg of pre epoch
        if args.early_stop:
            pre = 50
            if epoch >= pre*2 and sum(history_train_loss[epoch-pre:epoch])/pre >= sum(history_train_loss[epoch-pre*2:epoch-pre])/pre:
                print(f"Early Stop At Epoch {epoch}")
                break

    OUTPUT_FOLDER = './weight'
    WEIGHT_NAME = args.weight if args.weight else f'blstm_weight_s{N}_e{epoch}_p{N_PHYSICS_DATA}'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER,WEIGHT_NAME))

    fig, ax = plt.subplots()
    ax.plot(range(1, len(history_train_loss)+1) , history_train_loss)
    ax.set_title('BLSTM')
    ax.set_ylabel('Train Loss')
    ax.set_xlabel('Epoch')
    plt.show()
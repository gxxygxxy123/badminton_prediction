import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import csv
import numpy as np
import time
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from tqdm import trange
import argparse
import scipy.interpolate
import sys
sns.set()

from dataloader import RNNDataSet, PhysicsDataSet_seq2seq
from seq2seq import Encoder, Decoder, Seq2Seq

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip, device):
    
    model.train()
    
    epoch_loss = 0

    for batch_idx, (data, label, output_fps) in enumerate(iterator):
        src = data.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, N, IN_SIZE]
        trg = label.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, trg_len, OUT_SIZE]

        optimizer.zero_grad()

        output = model(src, trg)
        #output = [BATCH_SIZE, trg_len, OUT_SIZE]

        # output_dim = output.shape[-1]

        # output = output[:,1:,:].view(-1, output_dim)
        # trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(iterator):
            src = data.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, N, IN_SIZE]
            trg = label.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, trg_len, OUT_SIZE]

            output = model(src, trg, 0) #turn off teacher forcing
            #output = [BATCH_SIZE, trg_len, OUT_SIZE]

            output_dim = output.shape[-1]
            
            # output = output[1:].view(-1, output_dim)
            # trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
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
    parser = argparse.ArgumentParser(description="Seq2Seq Training Program")
    # parser.add_argument("-s","--seq", type=int, help="Input Time", required=True)
    parser.add_argument("-e","--epoch", type=int, help="Training Epochs", required=True)
    parser.add_argument("--in_dropout", type=float, help="Input dropout", default=0.0)
    parser.add_argument("--hidden_size", type=int, help="Hidden Size", default=16)
    parser.add_argument("--hidden_layer", type=int, help="Hidden Layer", default=2)
    parser.add_argument("--physics_data", type=int, help="Training Datas", default=140000)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=1)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.01) # SGD default lr=0.01
    parser.add_argument("-w","--weight", type=str, help="Ouput Weight name")
    parser.add_argument("--save_epoch", type=int, help="Save at each N epoch", default=100)
    parser.add_argument('--early_stop', action="store_true", help = 'Early Stop')
    args = parser.parse_args()

    # N = args.seq # Time Sequence Number
    N_EPOCHS = args.epoch
    IN_DROPOUT = args.in_dropout
    HIDDEN_SIZE = args.hidden_size
    N_LAYERS = args.hidden_layer
    N_PHYSICS_DATA = args.physics_data
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    SAVE_EPOCH = args.save_epoch

    print(#f"Input Seq: {N}\n"
          f"Epoch: {N_EPOCHS}\n"
          f"Input Dropout: {IN_DROPOUT}\n"
          f"Hidden Size: {HIDDEN_SIZE}\n"
          f"Hidden Layer: {N_LAYERS}\n"
          f"Physics data: {N_PHYSICS_DATA}\n"
          f"Batch Size: {BATCH_SIZE}\n"
          f"Learning Rate: {LEARNING_RATE}\n"
          f"Save At Each {SAVE_EPOCH} epoch\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train Dataset
    train_dataset = PhysicsDataSet_seq2seq(datas=N_PHYSICS_DATA)
    train_dataset_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)

    # Valid Dataset
    valid_dataset_dataloader_list = []
    for n in range(12,13):
        print(f"Valid n: {n}")
        valid_dataset_dataloader_list.append(
            torch.utils.data.DataLoader(dataset = RNNDataSet(dataset_path="../trajectories_dataset/valid/", 
            fps=120, N=n, move_origin_2d=False, smooth_2d=True, network='seq2seq'), batch_size = BATCH_SIZE, shuffle = True, drop_last=True))

    INPUT_DIM = 3 # X Y t
    OUTPUT_DIM = 3 # X Y t
    # ENC_EMB_DIM = 8
    # DEC_EMB_DIM = 8
    # ENC_DROPOUT = 0.5
    # DEC_DROPOUT = 0.5

    CLIP = 1

    enc = Encoder(INPUT_DIM, HIDDEN_SIZE, N_LAYERS, in_dropout=IN_DROPOUT)
    dec = Decoder(OUTPUT_DIM, INPUT_DIM, HIDDEN_SIZE, N_LAYERS)
    
    model = Seq2Seq(enc, dec, device)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    model.to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.MSELoss(reduction='mean')
    # criterion = loss_function
    # print("My loss function")

    history_train_loss = []

    # Early stopping
    last_valid_loss = float('inf')
    patience = 5
    triggertimes = 0

    print(f"Start Training ... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for epoch in range(1,N_EPOCHS+1):

        train_loss = train(model, train_dataset_dataloader, optimizer, criterion, CLIP, device=device)
        history_train_loss.append(train_loss)

        valid_loss = []
        for v in valid_dataset_dataloader_list:
            valid_loss.append(evaluate(model, v, criterion, device=device))
        valid_loss = sum(valid_loss)/len(valid_loss)

        print(f"Epoch: {epoch}/{N_EPOCHS}. Train Loss: {train_loss:.4f}. Valid Loss: {valid_loss:.4f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        if epoch % SAVE_EPOCH == 0:
            # print(f"Epoch: {epoch}/{N_EPOCHS}. Loss: {train_loss:.4f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            torch.save(model.state_dict(), f'./weight/seq2seq_weight_p{N_PHYSICS_DATA}_e{epoch}')
            print(f"Save weight ./weight/seq2seq_weight_p{N_PHYSICS_DATA}_e{epoch}")

        # Early Stopping
        if args.early_stop:
            if valid_loss > last_valid_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    torch.save(model.state_dict(), f'./weight/seq2seq_weight_p{N_PHYSICS_DATA}_best')
                    print(f"Early Stop At Epoch {epoch}. Save weight ./weight/seq2seq_weight_p{N_PHYSICS_DATA}_best")
                    break
            else:
                trigger_times = 0
            last_valid_loss = valid_loss
                

    OUTPUT_FOLDER = './weight'
    WEIGHT_NAME = args.weight if args.weight else f'seq2seq_weight_p{N_PHYSICS_DATA}_e{epoch}'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER,WEIGHT_NAME))

    fig, ax = plt.subplots()
    ax.plot(range(1, len(history_train_loss)+1) , history_train_loss)
    ax.set_title('Seq2Seq')
    ax.set_ylabel('Train Loss')
    ax.set_xlabel('Epoch')

    plt.show()
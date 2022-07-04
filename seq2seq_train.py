from code import interact
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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import argparse
import scipy.interpolate
from predict import predict2d
import sys
sns.set()

from dataloader import RNNDataSet, PhysicsDataSet
from seq2seq import Attention, Encoder, Decoder, Seq2Seq

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.01, 0.01)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip, device):
    
    model.train()
    
    epoch_loss = 0

    for batch_idx, (src, src_lens, trg, trg_lens, output_fps) in enumerate(iterator):

        src = src.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, N, IN_SIZE]
        trg = trg.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, trg_len, OUT_SIZE]

        optimizer.zero_grad()


        output = model(src, src_lens, trg, trg_lens, output_fps)
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

def evaluate(model, iterator, criterion, device, epoch, N, fps):

    model.eval()
    
    valid_loss = 0
    infer_loss = 0
    infer_i = 0

    fig,ax = plt.subplots()
    ax.set_title(f"Epoch {epoch}")

    #fig_dt, ax_dt = plt.subplots()
    
    # dt = []

    with torch.no_grad():
        # for batch_idx, (src, src_lens, trg, trg_lens, output_fps) in enumerate(iterator):
        #     if batch_idx > 0:
        #         continue

        #     src = src.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, N, IN_SIZE]
        #     trg = trg.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, trg_len, OUT_SIZE]
        #     output = model(src, src_lens, trg, trg_lens, output_fps, 0) #turn off teacher forcing

        #     #output = [BATCH_SIZE, trg_len, OUT_SIZE]

        #     loss = criterion(output, trg)
            
        #     valid_loss += loss.item()

        if epoch % 1 == 0:
            os.makedirs(args.fig_path, exist_ok=True)
            fig.savefig(os.path.join(args.fig_path, f'{epoch}_{N}.png'),dpi=200)
            ax.clear()
            #fig_dt.savefig(os.path.join(args.fig_path, f'{epoch}_dt.png'))
    plt.close(fig)
    #plt.close(fig_dt)

    return valid_loss / len(iterator), infer_loss / infer_i # TODO conbine valid & inference


def inference(model, dataset, criterion, device, epoch, N, fps):

    model.eval()

    infer_loss = 0
    infer_i = 0

    fig,ax = plt.subplots()
    ax.set_title(f"Epoch {epoch}")

    #fig_dt, ax_dt = plt.subplots()
    
    # dt = []

    with torch.no_grad():

        for idx, trajectory in dataset.whole_2d().items():
            if trajectory.shape[0] < N:
                continue

            ### debug
            # trajectory[:,[0,1]] -= trajectory[N-1,[0,1]]

            output_2d = predict2d(trajectory[:N], model, 'seq2seq', touch_ground_stop=False, input_fps=None, output_fps=fps, output_time=2.0, device=device)

            p = ax.plot(trajectory[:N,0], trajectory[:N,1], marker='o', markersize=1)
            ax.plot(trajectory[N-1:,0], trajectory[N-1:,1], color=p[0].get_color(), linestyle='--')
            ax.plot(output_2d[N:,0], output_2d[N:,1], alpha=0.3, color=p[0].get_color())

            # dt_diff = np.diff(output_2d[N:,-1],axis=0)
            # dt_diff = dt_diff[~np.isnan(dt_diff)]
            # dt += dt_diff.tolist()

            #ax_dt.hist(dt, color='red')
            #ax_dt.set_title(f"dt (Transformer) mean{sum(dt)/len(dt):.5f}")

            loss = criterion(torch.from_numpy(output_2d[N:trajectory.shape[0]]), torch.from_numpy(trajectory[N:]))
        
            infer_loss += loss.item()

            infer_i += 1

        if epoch % 1 == 0:
            os.makedirs(args.fig_path, exist_ok=True)
            fig.savefig(os.path.join(args.fig_path, f'{epoch}_{N}.png'),dpi=200)
            ax.clear()
            #fig_dt.savefig(os.path.join(args.fig_path, f'{epoch}_dt.png'))
    plt.close(fig)
    #plt.close(fig_dt)

    return infer_loss / infer_i



def collate_fn(batch):
    (src, trg, output_fps) = zip(*batch)
    src, trg, output_fps = list(src), list(trg), list(output_fps)

    src = sorted(src, key=lambda x: len(x), reverse=True)
    src_lens = [len(x) for x in src]

    src_pad = pad_sequence(src, batch_first=True, padding_value=0)

    trg = sorted(trg, key=lambda x: len(x), reverse=True)
    trg_lens = [len(x) for x in trg]
    trg_pad = pad_sequence(trg, batch_first=True, padding_value=0)

    return src_pad, src_lens, trg_pad, trg_lens, output_fps[0]

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target):

        # mask is target where dx/dy/dt != 0
        mask = (target != 0.0)
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result


class CumMSELoss(torch.nn.Module):
    def __init__(self):
        super(CumMSELoss, self).__init__()

    def forward(self, input, target):

        # mask is target where dx/dy != 0
        mask = (target != 0.0)

        input = torch.cumsum(input,dim=1)
        target = torch.cumsum(target,dim=1)

        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Seq2Seq Training Program")
    parser.add_argument("-e","--epoch", type=int, help="Training Epochs", required=True)
    parser.add_argument("--enc_dropout", type=float, help="Encoder dropout", default=0.5)
    parser.add_argument("--dec_dropout", type=float, help="Decoder dropout", default=0.5)
    parser.add_argument("--enc_hid_dim", type=int, help="Encoder Hidden Size", default=32)
    parser.add_argument("--dec_hid_dim", type=int, help="Decoder Hidden Size", default=32)
    parser.add_argument("--hidden_layer", type=int, help="Hidden Layer", default=1)
    parser.add_argument("--physics_data", type=int, help="Training Datas", default=140000)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001) # Adam default lr=0.001
    parser.add_argument("-w","--weight", type=str, help="Ouput Weight name")
    parser.add_argument("--save_epoch", type=int, help="Save at each N epoch", default=100)
    parser.add_argument('--early_stop', action="store_true", help = 'Early Stop')
    parser.add_argument('--fig_path', type=str)
    args = parser.parse_args()

    # N = args.seq # Time Sequence Number
    N_EPOCHS = args.epoch
    ENC_DROPOUT = args.enc_dropout
    DEC_DROPOUT = args.dec_dropout
    ENC_HID_DIM = args.enc_hid_dim
    DEC_HID_DIM = args.dec_hid_dim
    N_LAYERS = args.hidden_layer
    N_PHYSICS_DATA = args.physics_data
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    SAVE_EPOCH = args.save_epoch

    print(#f"Input Seq: {N}\n"
          f"Epoch: {N_EPOCHS}\n"
          f"Encoder Dropout: {ENC_DROPOUT}\n"
          f"Decoder Dropout: {DEC_DROPOUT}\n"
          f"Encoder Hidden Size: {ENC_HID_DIM}\n"
          f"Decoder Hidden Size: {DEC_HID_DIM}\n"
          f"Hidden Layer: {N_LAYERS}\n"
          f"Physics data: {N_PHYSICS_DATA}\n"
          f"Batch Size: {BATCH_SIZE}\n"
          f"Learning Rate: {LEARNING_RATE}\n"
          f"Save At Each {SAVE_EPOCH} epoch\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train Dataset
    train_dataset = PhysicsDataSet(datas=N_PHYSICS_DATA)
    #train_dataset = RNNDataSet(dataset_path="../trajectories_dataset/valid/", fps=120, N=24, smooth_2d=True, network='seq2seq') #debug
    #print("Debug Train dataset ours!")
    train_dataset_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn, drop_last=True)

    # Inference Dataset
    infer_dataset_list = []
    fpsss = [120]
    nnn = [6,12,18,24]

    print(f"Inference first {nnn} points... Output FPS {fpsss}")
    for fps in (fpsss):
        infer_dataset_list.append(RNNDataSet(dataset_path="../trajectories_dataset/valid/", 
                        fps=fps, smooth_2d=True, network='seq2seq'))


    INPUT_DIM = 2 # X Y t
    OUTPUT_DIM = 2 # X Y t

    CLIP = 1

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, N_LAYERS, DEC_HID_DIM, dropout=ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, attn, dropout=DEC_DROPOUT)
    
    model = Seq2Seq(enc, dec, device)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    model.to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # criterion = CumMSELoss()
    criterion = MaskedMSELoss()
    print(f"criterion: {criterion.__class__.__name__}")

    history_train_loss = []

    # Early stopping
    last_valid_loss = float('inf')
    patience = 5
    triggertimes = 0

    print(f"Start Training ... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for epoch in range(1,N_EPOCHS+1):

        train_loss = train(model, train_dataset_dataloader, optimizer, criterion, CLIP, device=device)
        history_train_loss.append(train_loss)

        infer_loss = []
        for dset in infer_dataset_list:
            for n in nnn:
                i_loss = inference(model, dset, criterion, device=device, epoch=epoch, N=n, fps=fps)
                infer_loss.append(i_loss)

        infer_loss = sum(infer_loss)/len(infer_loss)
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}/{N_EPOCHS}. Train Loss: {train_loss:.8f}. Infer Loss: {infer_loss:.8f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        if epoch % SAVE_EPOCH == 0:
            # print(f"Epoch: {epoch}/{N_EPOCHS}. Loss: {train_loss:.8f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            torch.save(model.state_dict(), f'./weight/seq2seq_weight_p{N_PHYSICS_DATA}_e{epoch}')
            print(f"Save weight ./weight/seq2seq_weight_p{N_PHYSICS_DATA}_e{epoch}")

        # Early Stopping
        if args.early_stop:
            if valid_loss > last_valid_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early Stop At Epoch {epoch}")
                    break
            else:
                trigger_times = 0
                torch.save(model.state_dict(), f'./weight/seq2seq_weight_p{N_PHYSICS_DATA}_best')
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
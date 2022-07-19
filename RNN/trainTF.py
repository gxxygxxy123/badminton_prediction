import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import csv
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import argparse
import sys

sns.set()
import individual_TF
from utils import predict, param, error_function
from dataloader import RNNDataSet, PhysicsDataSet

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")

from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt





def train(model, iterator, optimizer, criterion, mean, std, device):

    model.train()

    epoch_loss = 0

    for batch_idx, batch in enumerate(iterator):

        optimizer.optimizer.zero_grad()

        inp=(batch['src'][:,:].to(device)-mean.to(device))/std.to(device)
        target=(batch['trg'][:,:-1].to(device)-mean.to(device))/std.to(device)

        target_c = torch.zeros((target.shape[0],target.shape[1],1)).to(device)
        target = torch.cat((target,target_c),-1)
        start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)

        dec_inp = torch.cat((start_of_seq, target), 1)

        src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)

        pred = model(inp, dec_inp, src_att, trg_att)

        loss = criterion(pred[:, :,0:2]
        .contiguous().view(-1, 2), ((batch['trg'][:, :, :]
        .to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)
                        ).mean()# + torch.mean(torch.abs(pred[:,:,2])) this is unneccessary
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def inference(model, dataset, criterion, mean, std, device, epoch, N):

    model.eval()

    infer_loss = []

    fig,ax = plt.subplots()
    ax.set_title(f"Epoch {epoch}, input {N} points, 120 FPS")
    ax.set_xlabel(f"distance (m)")
    ax.set_ylabel(f"height (m)")

    with torch.no_grad():

        for idx, trajectory in dataset.whole_2d().items():
            if trajectory.shape[0] < N:
                continue

            inp = trajectory[:N].copy()
            gt = trajectory[N:].copy()

            # TODO test missing values, fill dropped point x,y as 0 (pad) XXXXX
            #inp[1:-1:2, [0,1]] = 0

            out = predict.predict2d_TF(inp, model, mean=mean, std=std, out_time=3.0, fps=120.0, touch_ground_stop=True, device=device)

            p = ax.plot(inp[:,0], inp[:,1], marker='o', markersize=1)
            ax.plot(gt[:,0], gt[:,1], color=p[0].get_color(), linestyle='--')
            ax.plot(out[inp.shape[0]:,0], out[inp.shape[0]:,1], marker='o', markersize=1, alpha=0.3, color=p[0].get_color())

            ax.scatter(gt[::10,0],gt[::10,1], color='red',s=2)
            ax.scatter(out[inp.shape[0]::10,0],out[inp.shape[0]::10,1], color='blue',s=2)

            infer_loss.append(error_function.space_time_err(gt,out[inp.shape[0]:]))

        if epoch % 1 == 0:
            fig.savefig(os.path.join(args.fig_pth, f'{epoch}_{N}.png'),dpi=200)
            ax.clear()

    plt.close(fig)

    return sum(infer_loss) / len(infer_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Training Program")
    parser.add_argument("-e","--epoch", type=int, help="Training Epochs", default=500)
    parser.add_argument("--physics_data", type=int, help="Training Datas", default=30000)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=32)
    parser.add_argument("--save_epoch", type=int, help="Save at each N epoch", default=100)
    parser.add_argument('--early_stop', action="store_true", help = 'Early Stop')
    parser.add_argument('--fig_pth', type=str, default='./figure/TF/')
    parser.add_argument('--wgt_pth', type=str, default='./weight/TF/')

    parser.add_argument('--obs',type=int,default=24) # TO DELETE
    parser.add_argument('--preds',type=int,default=360) # TO DELETE
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--factor', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=10)


    args = parser.parse_args()

    os.makedirs(args.fig_pth, exist_ok=True)
    os.makedirs(args.wgt_pth, exist_ok=True)


    print(f"Epoch: {args.epoch}\n"
          f"Physics data: {args.physics_data}\n"
          f"Batch Size: {args.batch_size}\n"
          f"Save At Each {args.save_epoch} epoch\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train Dataset
    train_dataset = PhysicsDataSet(datas=args.physics_data, model='TF', dxyt=True, network_in_dim=2, drop_mode=0)
    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last=True)

    # Inference Dataset
    infer_dataset_list = []
    fpsss = [120]
    nnn = [24]

    print(f"Inference first {nnn} points... Output FPS {fpsss}")
    for fps in (fpsss):
        infer_dataset_list.append(RNNDataSet(dataset_path="../trajectories_dataset/valid/", 
                        fps=fps, smooth_2d=True, network='TF'))

    model=individual_TF.IndividualTF(2, 3, 3, N=args.layers,
                   d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)
    print(f'The model has {param.count_parameters(model):,} trainable parameters')
    #model.apply(param.init_weights)

    optimizer = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
                        optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    criterion = F.pairwise_distance

    mean = train_dataset.mean()
    std = train_dataset.std()

    hist_tr_loss = []

    # Early stopping
    last_val_loss = float('inf')
    patience = 5
    triggertimes = 0

    print(f"Start Training ... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for epoch in range(1,args.epoch+1):

        tr_loss = train(model, tr_dl, optimizer, criterion, mean=mean, std=std, device=device)
        hist_tr_loss.append(tr_loss)

        infer_loss = []
        for dset in infer_dataset_list:
            for n in nnn:
                i_loss = inference(model, dset, criterion, mean=mean, std=std, device=device, epoch=epoch, N=n)
                infer_loss.append(i_loss)
        infer_loss = sum(infer_loss)/len(infer_loss)

        print(f"Epoch: {epoch}/{args.epoch}. Train Loss: {tr_loss:.8f}. Infer Loss: {infer_loss:.8f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        if epoch % args.save_epoch == 0:
            torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.optimizer.state_dict(),
                        'datas':args.physics_data,
                        'mean':mean,
                        'std':std
                       },  os.path.join(args.wgt_pth, f'TF_p{args.physics_data}_e{epoch}'))
            print(f"Save Weight At Epoch {epoch}")

        # Early Stopping TODO
        if args.early_stop:
            if valid_loss > last_val_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early Stop At Epoch {epoch}")
                    break
            else:
                trigger_times = 0
                torch.save(model.state_dict(),     f'./weight/transformer_weight_p{args.physics_data}_best')
            last_val_loss = valid_loss
                


    torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.optimizer.state_dict(),
                'datas':args.physics_data,
                'mean':mean,
                'std':std
                },  os.path.join(args.wgt_pth, f'TF_final'))

    fig, ax = plt.subplots()
    ax.plot(range(1, len(hist_tr_loss)+1) , hist_tr_loss)
    ax.set_title('Transformer')
    ax.set_ylabel('Train Loss')
    ax.set_xlabel('Epoch')
    plt.show()
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import csv
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import argparse
import sys
import predict
sns.set()

from dataloader import RNNDataSet, PhysicsDataSet
from transformer import TimeSeriesTransformer

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, src_mask, trg_mask, device):
    
    model.train()
    
    epoch_loss = 0

    for batch_idx, (src, trg, trg_y) in enumerate(iterator):

        src = src.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, N, IN_SIZE]
        trg = trg.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, trg_len, OUT_SIZE]
        trg_y = trg_y.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, trg_len, OUT_SIZE]

        optimizer.zero_grad()

        output = model(src, trg, src_mask, trg_mask)

        #output = [BATCH_SIZE, trg_len, OUT_SIZE]

        loss = criterion(output, trg_y)

        loss.backward()

        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, src_mask, trg_mask, device, epoch):
    
    model.eval()
    
    epoch_loss = 0

    fig,ax = plt.subplots()
    ax.set_title(f"Epoch {epoch}")

    fig_dt, ax_dt = plt.subplots()
    
    dt = []

    with torch.no_grad():
        for batch_idx, (src, trg, output_fps) in enumerate(iterator):
            src = src.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, N, IN_SIZE]
            trg = trg.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, trg_len, OUT_SIZE]

            output = model(src, trg, src_mask, trg_mask)

            #output = [BATCH_SIZE, trg_len, OUT_SIZE]

            output_dim = output.shape[-1]
            
            # output = output[1:].view(-1, output_dim)
            # trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()

        N = iterator.dataset.N
        for idx, trajectory in iterator.dataset.whole_2d().items():
            if trajectory.shape[0] < N:
                continue
            output_2d = predict.predict2d(trajectory[:N], model, 'transformer', device=device)

            p = ax.plot(trajectory[:,0],trajectory[:,1],marker='o',markersize=1)
            ax.plot(output_2d[:,0],output_2d[:,1],marker='o',markersize=1,alpha=0.3,color=p[0].get_color(), linestyle='--')

            dt_diff = np.diff(output_2d[N:,-1],axis=0)
            dt_diff = dt_diff[~np.isnan(dt_diff)]
            dt += dt_diff.tolist()
        if len(dt) > 0:
            ax_dt.hist(dt, color='red')
            ax_dt.set_title(f"dt (Transformer) mean{sum(dt)/len(dt):.5f}")

    if epoch % 10 == 0:
        os.makedirs(args.fig_path, exist_ok=True)
        fig.savefig(os.path.join(args.fig_path, f'{epoch}.png')) 
        fig_dt.savefig(os.path.join(args.fig_path, f'{epoch}_dt.png'))
    plt.close(fig)
    plt.close(fig_dt)

    return epoch_loss / len(iterator)


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

        self.debug_rm_ts = False
        print(f"MaskedMSELoss, debug {self.debug_rm_ts}")

    def forward(self, input, target):

        # Debug remove ts !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.debug_rm_ts:
            input = input[:,:,:-1]
            target = target[:,:,:-1]

        # mask is target where dx/dy/dt != 0
        mask = (target != 0.0)

        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result

def generate_square_subsequent_mask(dim1: int, dim2: int, dim3: int, device=torch.device('cpu')):
    return torch.triu(torch.ones(dim1, dim2, dim3) * float('-inf'), diagonal=1).to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Training Program")
    parser.add_argument("-e","--epoch", type=int, help="Training Epochs", required=True)
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
    N_PHYSICS_DATA = args.physics_data
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    SAVE_EPOCH = args.save_epoch



    print(#f"Input Seq: {N}\n"
          f"Epoch: {N_EPOCHS}\n"
          f"Physics data: {N_PHYSICS_DATA}\n"
          f"Batch Size: {BATCH_SIZE}\n"
          f"Learning Rate: {LEARNING_RATE}\n"
          f"Save At Each {SAVE_EPOCH} epoch\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    output_fps = 120

    # Train Dataset
    train_dataset = PhysicsDataSet(datas=N_PHYSICS_DATA, model='transformer')
    train_dataset_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)


    # Valid Dataset
    valid_dataset_dataloader_list = []
    for fps in ([120]):
        for n in ([20]):
            valid_dataset_dataloader_list.append(
                torch.utils.data.DataLoader(dataset = RNNDataSet(dataset_path="../trajectories_dataset/valid/",
                fps=fps, N=n, move_origin_2d=False, smooth_2d=True, network='transformer'), batch_size = 1, shuffle = True, drop_last=True))
                # torch.utils.data.DataLoader(dataset= PhysicsDataSet_transformer(datas=100),batch_size = 1, shuffle = True, drop_last=True))

    dim_val = 32 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 2 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 2 # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 2 # Number of times the encoder layer is stacked in the encoder
    input_size = 3 # The number of input variables. 1 if univariate forecasting.
    dec_seq_len = 1000 # length of input given to decoder. Can have any integer value.
    enc_seq_len = 100 # length of input given to encoder. Can have any integer value.
    output_sequence_length = 1000 # Length of the target sequence, i.e. how many time steps should your forecast cover
    max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder



    # Mask
    src_mask = generate_square_subsequent_mask(
    dim1=BATCH_SIZE*n_heads,
    dim2=output_sequence_length,
    dim3=enc_seq_len,
    device=device
    )

    trg_mask = generate_square_subsequent_mask( 
        dim1=BATCH_SIZE*n_heads,
        dim2=output_sequence_length,
        dim3=output_sequence_length,
        device=device
        )

    # Eva Mask
    src_eva_mask = generate_square_subsequent_mask(
    dim1=1*n_heads,
    dim2=output_sequence_length,
    dim3=enc_seq_len,
    device=device
    )

    trg_eva_mask = generate_square_subsequent_mask( 
        dim1=1*n_heads,
        dim2=output_sequence_length,
        dim3=output_sequence_length,
        device=device
        )



    print("enc_seq_len:",enc_seq_len)
    print("output_sequence_length",output_sequence_length)

    model = TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size, 
        dec_seq_len=dec_seq_len,
        max_seq_len=max_seq_len,
        out_seq_len=output_sequence_length, 
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    model.to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = MaskedMSELoss()

    history_train_loss = []

    # Early stopping
    last_valid_loss = float('inf')
    patience = 5
    triggertimes = 0

    print(f"Start Training ... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for epoch in range(1,N_EPOCHS+1):

        train_loss = train(model, train_dataset_dataloader, optimizer, criterion, src_mask, trg_mask, device=device)
        history_train_loss.append(train_loss)

        valid_loss = []
        for v in valid_dataset_dataloader_list:
            valid_loss.append(evaluate(model, v, criterion, src_eva_mask, trg_eva_mask, device=device, epoch=epoch))
        valid_loss = sum(valid_loss)/len(valid_loss)

        print(f"Epoch: {epoch}/{N_EPOCHS}. Train Loss: {train_loss:.8f}. Valid Loss: {valid_loss:.8f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        if epoch % SAVE_EPOCH == 0:
            # print(f"Epoch: {epoch}/{N_EPOCHS}. Loss: {train_loss:.8f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            torch.save(model.state_dict(), f'./weight/transformer_weight_p{N_PHYSICS_DATA}_e{epoch}')
            print(f"Save weight ./weight/transformer_weight_p{N_PHYSICS_DATA}_e{epoch}")

        # Early Stopping
        if args.early_stop:
            if valid_loss > last_valid_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early Stop At Epoch {epoch}")
                    break
            else:
                trigger_times = 0
                torch.save(model.state_dict(), f'./weight/transformer_weight_p{N_PHYSICS_DATA}_best')
            last_valid_loss = valid_loss
                

    OUTPUT_FOLDER = './weight'
    WEIGHT_NAME = args.weight if args.weight else f'transformer_weight_p{N_PHYSICS_DATA}_e{epoch}'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER,WEIGHT_NAME))

    fig, ax = plt.subplots()
    ax.plot(range(1, len(history_train_loss)+1) , history_train_loss)
    ax.set_title('Transformer')
    ax.set_ylabel('Train Loss')
    ax.set_xlabel('Epoch')

    plt.show()
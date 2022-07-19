import torch
import pandas as pd
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
from datetime import datetime
from seq2seq import Seq2Seq
from blstm import Blstm
from seq2seq import Attention, Encoder, Decoder, Seq2Seq
from transformer import TimeSeriesTransformer
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from dataloader import RNNDataSet, PhysicsDataSet
from predict import predict3d, predict2d
from physic_model import physics_predict3d
from error_function import space_err, time_err, space_time_err, time_after_err
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from point import Point, load_points_from_csv, save_points_to_csv

parser = argparse.ArgumentParser(description="Trajectories Testing Program")
parser.add_argument("-s","--seq", type=int, help="BLSTM & Seq2Seq Input Sequence", default=10)
parser.add_argument("--blstm_weight", type=str, help="BLSTM Weight", default=None)
parser.add_argument("--seq2seq_weight", type=str, help="Seq2Seq Weight", default=None)
parser.add_argument("--transformer_weight", type=str, help="Transformer Weight", default=None)
parser.add_argument("--folder", type=str, help="Test Folder", required=True)
parser.add_argument('--no_show', action="store_true", help = 'No plt.show()')
parser.add_argument("--fps", type=float, default=None, help="Trajectories FPS")
args = parser.parse_args()

# Argument
N = args.seq
BLSTM_WEIGHT = args.blstm_weight
SEQ2SEQ_WEIGHT = args.seq2seq_weight
TRANSFORMER_WEIGHT = args.transformer_weight
TEST_DATASET = args.folder
fps = args.fps

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset
test_dataset = RNNDataSet(dataset_path=TEST_DATASET, fps=fps, N=N, smooth_2d=True, smooth_3d=False)
#test_dataset = PhysicsDataSet_seq2seq(datas=10)

trajectories_2d = test_dataset.whole_2d()
trajectories_3d = test_dataset.whole_3d()
trajectories_3d2d = test_dataset.whole_3d2d()

################### Method 1 ################# BLSTM
if BLSTM_WEIGHT:
    # Evaluation
    blstm_space_2d = []
    blstm_space_time_2d = []
    blstm_space_3d = []
    blstm_space_time_3d = []
    blstm_time = []
    blstm_time_after_3d = []

    HIDDEN_SIZE = 16
    N_LAYERS = 2

    model = Blstm(hidden_size=HIDDEN_SIZE, hidden_layer=N_LAYERS)
    model.load_state_dict(torch.load(BLSTM_WEIGHT))
    model.eval()

    fig1, ax1 = plt.subplots()
    ax1.set_title("BLSTM")

    with torch.no_grad():
        for idx, trajectory in trajectories_2d.items():
            if trajectory.shape[0] < N:
                continue

            output_2d = predict2d(trajectory[:N], model, 'blstm')

            p = ax1.plot(trajectory[:,0],trajectory[:,1],marker='o',markersize=2)
            ax1.plot(output_2d[:,0],output_2d[:,1],marker='o',markersize=4,alpha=0.3,color=p[0].get_color(), linestyle='--')

            blstm_space_2d.append(space_err(trajectory[N:], output_2d[N:]))
            blstm_space_time_2d.append(space_time_err(trajectory[N:], output_2d[N:]))
            blstm_time.append(time_err(trajectory[N:], output_2d[N:]))
            
        for idx, trajectory in trajectories_3d.items():
            if trajectory.shape[0] < N:
                continue

            output_3d = predict3d(trajectory[:N], model, 'blstm')

            blstm_space_3d.append(space_err(trajectory[N:], output_3d[N:]))
            blstm_space_time_3d.append(space_time_err(trajectory[N:], output_3d[N:]))
            tmp = []
            for t in np.arange(0, 3, 0.1):
                tmp.append(time_after_err(trajectory, output_3d, fps, fps, t))
            blstm_time_after_3d.append(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            blstm_time_error_3d = np.nanmean(np.array(blstm_time_after_3d), axis=0)

    print(f"===BLSTM===")
    print(f"Weight: {BLSTM_WEIGHT}")
    print(f"Space Error 2D (Average): {np.nanmean(blstm_space_2d):.2f}m")
    print(f"Space Time Error 2D (Average): {np.nanmean(blstm_space_time_2d):.2f}m")
    print(f"Space Error 3D (Average): {np.nanmean(blstm_space_3d):.2f}m")
    print(f"Space Time Error 3D (Average): {np.nanmean(blstm_space_time_3d):.2f}m")
    print(f"Time Error (Average): {np.nanmean(blstm_time):.3f}s")
    # ax2.plot(np.arange(0, 3, 0.1),time_after_error_3d)


################### Method 2 ################# Seq2Seq
if SEQ2SEQ_WEIGHT:
    # Evaluation
    seq2seq_space_2d = []
    seq2seq_space_time_2d = []
    seq2seq_space_3d = []
    seq2seq_space_time_3d = []
    seq2seq_time = []
    seq2seq_time_after_3d = []

    INPUT_DIM = 3 # X Y t
    OUTPUT_DIM = 3 # X Y t
    ENC_HID_DIM = 128
    DEC_HID_DIM = 64
    N_LAYERS = 1
    ENC_DROPOUT = 0.0
    DEC_DROPOUT = 0.0

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, N_LAYERS, DEC_HID_DIM, dropout=ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, attn, dropout=DEC_DROPOUT)

    model = Seq2Seq(encoder=enc, decoder=dec)
    model.load_state_dict(torch.load(SEQ2SEQ_WEIGHT))
    model.eval()

    fig2, ax2 = plt.subplots()
    ax2.set_title(f"Seq2Seq, FPS({fps}), N({N}), weight({SEQ2SEQ_WEIGHT})")

    fig_dt, ax_dt = plt.subplots()
    ax_dt.set_title("dt (Seq2Seq)")
    dt = []

    with torch.no_grad():
        for idx, trajectory in trajectories_2d.items():
            if trajectory.shape[0] < N:
                continue
            output_2d = predict2d(trajectory[:N], model, 'seq2seq', input_fps=None, output_fps=fps, output_time=3)

            p = ax2.plot(trajectory[:N,0], trajectory[:N,1], marker='o', markersize=1)
            ax2.plot(trajectory[N-1:,0], trajectory[N-1:,1], color=p[0].get_color(), linestyle='--')
            ax2.plot(output_2d[:,0], output_2d[:,1], alpha=0.3, color=p[0].get_color())

            dt += np.diff(output_2d[:,-1]).tolist()
            #seq2seq_space_2d.append(space_err(trajectory[N:], output_2d[N:]))
            #seq2seq_space_time_2d.append(space_time_err(trajectory[N:], output_2d[N:]))
            #seq2seq_time.append(time_err(trajectory[N:], output_2d[N:]))
        """
        for idx, trajectory in trajectories_3d.items():
            if trajectory.shape[0] < N:
                continue

            output_3d = predict3d(trajectory[:N], model, 'seq2seq', seq2seq_output_fps=test_dataset.fps())

            seq2seq_space_3d.append(space_err(trajectory[N:], output_3d[N:]))
            seq2seq_space_time_3d.append(space_time_err(trajectory[N:], output_3d[N:]))
            tmp = []
            for t in np.arange(0, 3, 0.1):
                tmp.append(time_after_err(trajectory, output_3d, fps, fps, t))
            seq2seq_time_after_3d.append(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            seq2seq_time_after_3d = np.nanmean(np.array(seq2seq_time_after_3d), axis=0)
        """
        ax_dt.hist(dt,color='red')
        print(sum(dt)/len(dt))
    # print(f"===Seq2Seq===")
    # print(f"Weight: {SEQ2SEQ_WEIGHT}")
    # print(f"Space Error 2D (Average): {np.nanmean(seq2seq_space_2d):.2f}m")
    # print(f"Space Time Error 2D (Average): {np.nanmean(seq2seq_space_time_2d):.2f}m")
    # print(f"Space Error 3D (Average): {np.nanmean(seq2seq_space_3d):.2f}m")
    # print(f"Space Time Error 3D (Average): {np.nanmean(seq2seq_space_time_3d):.2f}m")
    # print(f"Time Error (Average): {np.nanmean(seq2seq_time):.3f}s")

    # ax2.plot(np.arange(0, 3, 0.1),time_after_error_3d)



if not args.no_show:
    plt.show()

################### Method 3 ################# Transformer
if TRANSFORMER_WEIGHT:
    # Evaluation

    dim_val = 32 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 2 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 2 # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 2 # Number of times the encoder layer is stacked in the encoder
    input_size = 3 # The number of input variables. 1 if univariate forecasting.
    dec_seq_len = 1000 # length of input given to decoder. Can have any integer value.
    enc_seq_len = 100 # length of input given to encoder. Can have any integer value.
    output_sequence_length = 1000 # Length of the target sequence, i.e. how many time steps should your forecast cover
    max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder

    model = TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size, 
        dec_seq_len=dec_seq_len,
        max_seq_len=max_seq_len,
        out_seq_len=output_sequence_length, 
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads)
    model.load_state_dict(torch.load(TRANSFORMER_WEIGHT))
    model.eval()

    fig2, ax2 = plt.subplots()
    ax2.set_title(f"Seq2Seq, FPS({fps}), N({N}), weight({TRANSFORMER_WEIGHT})")

    fig_dt, ax_dt = plt.subplots()
    ax_dt.set_title("dt (Transformer)")
    dt = []

    with torch.no_grad():
        for idx, trajectory in trajectories_2d.items():
            if trajectory.shape[0] < N:
                continue
            output_2d = predict2d(trajectory[:N], model, 'transformer')

            p = ax2.plot(trajectory[:,0],trajectory[:,1],marker='o',markersize=1)
            ax2.plot(output_2d[:,0],output_2d[:,1],marker='o',markersize=1,alpha=0.3,color=p[0].get_color(), linestyle='--')

            dt += np.diff(output_2d[:,-1]).tolist()
            #seq2seq_space_2d.append(space_err(trajectory[N:], output_2d[N:]))
            #seq2seq_space_time_2d.append(space_time_err(trajectory[N:], output_2d[N:]))
            #seq2seq_time.append(time_err(trajectory[N:], output_2d[N:]))
        """
        for idx, trajectory in trajectories_3d.items():
            if trajectory.shape[0] < N:
                continue

            output_3d = predict3d(trajectory[:N], model, 'seq2seq', seq2seq_output_fps=test_dataset.fps())

            seq2seq_space_3d.append(space_err(trajectory[N:], output_3d[N:]))
            seq2seq_space_time_3d.append(space_time_err(trajectory[N:], output_3d[N:]))
            tmp = []
            for t in np.arange(0, 3, 0.1):
                tmp.append(time_after_err(trajectory, output_3d, fps, fps, t))
            seq2seq_time_after_3d.append(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            seq2seq_time_after_3d = np.nanmean(np.array(seq2seq_time_after_3d), axis=0)
        """
        ax_dt.hist(dt,color='red')
        print(sum(dt)/len(dt))
    # print(f"===Seq2Seq===")
    # print(f"Weight: {SEQ2SEQ_WEIGHT}")
    # print(f"Space Error 2D (Average): {np.nanmean(seq2seq_space_2d):.2f}m")
    # print(f"Space Time Error 2D (Average): {np.nanmean(seq2seq_space_time_2d):.2f}m")
    # print(f"Space Error 3D (Average): {np.nanmean(seq2seq_space_3d):.2f}m")
    # print(f"Space Time Error 3D (Average): {np.nanmean(seq2seq_space_time_3d):.2f}m")
    # print(f"Time Error (Average): {np.nanmean(seq2seq_time):.3f}s")

    # ax2.plot(np.arange(0, 3, 0.1),time_after_error_3d)



if not args.no_show:
    plt.show()

sys.exit(0)
################### Method 4 ################# Physics Model

# Evaluation
physics_space_2d = []
physics_space_time_2d = []
physics_space_3d = []
physics_space_time_3d = []
physics_time = []
physics_time_after_3d = []

for idx, trajectory in trajectories_3d2d.items():
    output_3d2d = physics_predict3d(trajectory[0,:], trajectory[1,:])

    # p = ax.plot(trajectory[:,0],trajectory[:,1],marker='o',markersize=4)
    # ax.plot(output_2d[:,0],output_2d[:,1],marker='o',markersize=4,alpha=0.3,color=p[0].get_color(), linestyle='--')
    physics_space_2d.append(space_err(trajectory[2:], output_3d2d[2:]))
    physics_space_time_2d.append(space_time_err(trajectory[2:], output_3d2d[2:]))

for idx, trajectory in trajectories_3d.items():
    output_3d = physics_predict3d(trajectory[0,:], trajectory[1,:])

    # p = ax3.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2],marker='o',markersize=4)
    # ax3.plot(output_3d[:,0],output_3d[:,1],output_3d[:,2],marker='o',markersize=4,alpha=0.3,color=p[0].get_color(), linestyle='--')


    physics_space_3d.append(space_err(trajectory[2:], output_3d[2:]))
    physics_space_time_3d.append(space_time_err(trajectory[2:], output_3d[2:]))
    physics_time.append(time_err(trajectory[2:], output_3d[2:]))
    
    tmp = []
    for t in np.arange(0, 3, 0.1):
        tmp.append(time_after_err(trajectory, output_3d, fps, fps, t))
    physics_time_after_3d.append(tmp)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    physics_time_after_3d = np.nanmean(np.array(physics_time_after_3d), axis=0)

print(f"===Physics===")
print(f"Space Error 2D (Average): {np.mean(physics_space_2d):.2f}m")
print(f"Space Time Error 2D (Average): {np.mean(physics_space_time_2d):.2f}m")
print(f"Space Error 3D (Average): {np.mean(physics_space_3d):.2f}m")
print(f"Space Time Error 3D (Average): {np.mean(physics_space_time_3d):.2f}m")
print(f"Time Error (Average): {np.mean(physics_time):.3f}s")
#ax2.plot(np.arange(0, 3, 0.1),physics_time_after_3d)













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
from seq2seq import Encoder, Decoder, Seq2Seq
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from dataloader import RNNDataSet, PhysicsDataSet_seq2seq
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
parser.add_argument("--folder", type=str, help="Test Folder", required=True)
parser.add_argument('--no_show', action="store_true", help = 'No plt.show()')
parser.add_argument("--fps", type=int, help="Trajectories FPS", required=True)
args = parser.parse_args()

# Argument
N = args.seq
BLSTM_WEIGHT = args.blstm_weight
SEQ2SEQ_WEIGHT = args.seq2seq_weight
TEST_DATASET = args.folder
fps = args.fps

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset
test_dataset = RNNDataSet(dataset_path=TEST_DATASET, N=N, smooth_2d=False, smooth_3d=False, fps=fps)
#test_dataset = PhysicsDataSet_seq2seq(datas=100, out_max_time=2)

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
    HIDDEN_SIZE = 16
    N_LAYERS = 2
    IN_DROPOUT = 0.0

    enc = Encoder(INPUT_DIM, HIDDEN_SIZE, N_LAYERS, in_dropout=IN_DROPOUT)
    dec = Decoder(OUTPUT_DIM, INPUT_DIM, HIDDEN_SIZE, N_LAYERS)

    model = Seq2Seq(encoder=enc, decoder=dec)
    model.load_state_dict(torch.load(SEQ2SEQ_WEIGHT))
    model.eval()

    fig2, ax2 = plt.subplots()
    ax2.set_title("Seq2Seq")

    with torch.no_grad():
        for idx, trajectory in trajectories_2d.items():
            if trajectory.shape[0] < N:
                continue
            output_2d = predict2d(trajectory[:N], model, 'seq2seq', seq2seq_output_fps=120)

            p = ax2.plot(trajectory[:,0],trajectory[:,1],marker='o',markersize=2)
            ax2.plot(output_2d[:,0],output_2d[:,1],marker='o',markersize=4,alpha=0.3,color=p[0].get_color(), linestyle='--')


            seq2seq_space_2d.append(space_err(trajectory[N:], output_2d[N:]))
            seq2seq_space_time_2d.append(space_time_err(trajectory[N:], output_2d[N:]))
            seq2seq_time.append(time_err(trajectory[N:], output_2d[N:]))
            
        for idx, trajectory in trajectories_3d.items():
            if trajectory.shape[0] < N:
                continue

            output_3d = predict3d(trajectory[:N], model, 'seq2seq', seq2seq_output_fps=120)

            seq2seq_space_3d.append(space_err(trajectory[N:], output_3d[N:]))
            seq2seq_space_time_3d.append(space_time_err(trajectory[N:], output_3d[N:]))
            tmp = []
            for t in np.arange(0, 3, 0.1):
                tmp.append(time_after_err(trajectory, output_3d, fps, fps, t))
            seq2seq_time_after_3d.append(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            seq2seq_time_after_3d = np.nanmean(np.array(seq2seq_time_after_3d), axis=0)

    print(f"===Seq2Seq===")
    print(f"Weight: {SEQ2SEQ_WEIGHT}")
    print(f"Space Error 2D (Average): {np.nanmean(seq2seq_space_2d):.2f}m")
    print(f"Space Time Error 2D (Average): {np.nanmean(seq2seq_space_time_2d):.2f}m")
    print(f"Space Error 3D (Average): {np.nanmean(seq2seq_space_3d):.2f}m")
    print(f"Space Time Error 3D (Average): {np.nanmean(seq2seq_space_time_3d):.2f}m")
    print(f"Time Error (Average): {np.nanmean(seq2seq_time):.3f}s")
    # ax2.plot(np.arange(0, 3, 0.1),time_after_error_3d)



if not args.no_show:
    plt.show()

sys.exit(0)
################### Method 3 ################# Physics Model

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













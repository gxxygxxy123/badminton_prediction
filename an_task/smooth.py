import argparse
import json
import logging
import os
import sys
import threading
import time
import csv
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import math
sns.set_style("white")

sys.path.append(f"../lib")
sys.path.append(f"../Model3D")
sys.path.append(f"../RNN")
from point import Point, load_points_from_csv, save_points_to_csv
from error_function import space_err, time_err, space_time_err
from physic_model import physics_predict3d
from threeDprojectTo2D import FitVerticalPlaneTo2D
from dataloader import RNNDataSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'smooth way comparison')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    parser.add_argument('--by', type=str, required=True, help = 'point/trajectory')
    args = parser.parse_args()
    assert args.by == 'point' or args.by == 'trajectory', "args.by wrong"

    # Dataset
    DATASET = args.folder # '../trajectories_dataset/split'
    N = 1

    # boxplot
    box = {}

    non_smooth_dataset = RNNDataSet(dataset_path=DATASET, N=N, smooth_2d=False, poly=-1)

    for poly in range(8,9):

        dis_error = []

        dataset = RNNDataSet(dataset_path=DATASET, N=N, smooth_2d=True, poly=poly)
        
        for idx, data in dataset.whole_2d().items():
            data2 = non_smooth_dataset.whole_2d()[idx].copy()
            assert data2.shape == data.shape, "Error"
            data -= data[0] # move to origin, reset time to zero
            data2 -= data2[0] # move to origin, reset time to zero

            plt.plot(data[:,0],data[:,1],marker='o',markersize=1, color='red')
            plt.plot(data2[:,0],data2[:,1],marker='o',markersize=1, color='blue')
            # plt.title(f"idx: {idx}")

            tmp = np.linalg.norm(data[:,:-1]-data2[:,:-1], axis=1).tolist()

            if args.by == 'point':
                dis_error = dis_error + tmp
            elif args.by == 'trajectory':
                dis_error.append(sum(tmp)/len(tmp))

        box[f"Poly {poly}"] = dis_error

        # plt.title(f"Poly fit {poly}")
        # plt.show()

    fig, ax = plt.subplots()
    ax.boxplot(box.values(), sym='')
    ax.set_xticklabels(box.keys())
    ax.set_ylabel("Error(m)")
    ax.set_title(f"After polynomial fitting. unit: {args.by}")

    plt.show()

    ### Foreach N points, draw it's smooth influence
    N = 5
    ITER = 12
    # boxplot
    poly = 8
    box = {}

    dataset = RNNDataSet(dataset_path=DATASET, N=N, smooth_2d=True, poly=poly)
    
    for idx, data in dataset.whole_2d().items():
        if args.by == 'trajectory' and data.shape[0] < N*ITER:
            continue
        data2 = non_smooth_dataset.whole_2d()[idx].copy()
        assert data2.shape == data.shape, "Error"
        data -= data[0] # move to origin, reset time to zero
        data2 -= data2[0] # move to origin, reset time to zero

        for i in range(ITER):
            if i not in box.keys():
                box[i] = []

            tmp = np.linalg.norm(data[i*N:(i+1)*N,:-1]-data2[i*N:(i+1)*N,:-1], axis=1).tolist()

            if args.by == 'point':
                box[i] = box[i] + tmp
            elif args.by == 'trajectory':
                box[i].append(sum(tmp)/len(tmp))

    fig, ax = plt.subplots()
    ax.boxplot(box.values(), sym='')
    ax.set_xticklabels([f"{s*N+1}~{(s+1)*N}" for s in box.keys()])
    ax.set_xlabel("N-th point")
    ax.set_ylabel("Distance(m)")
    ax.set_title(f"Smooth Poly {poly} influence by {args.by}")

    plt.show()


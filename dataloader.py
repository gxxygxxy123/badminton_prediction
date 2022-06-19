import torch.nn as nn
import torch
import pandas as pd
import os
import csv
import random
import numpy as np
import seaborn as sns
import math
import time
import sys
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
from scipy.integrate import solve_ivp

from scipy.signal import savgol_filter
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from physic_model import bm_ball

sys.path.append(f"../lib")
from point import Point, load_points_from_csv, save_points_to_csv, np2Point

class RNNDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset_path=None, N=5, move_origin_2d=True, smooth_2d=True, smooth_3d=False, fps=120, poly=8, network=None, csvfile='Model3D.csv'):

        # move_origin_2d: move the curve (self.input + self.ground_truth) to origin
        super(RNNDataSet).__init__()
        self.trajectories_2d = {}
        # self.trajectories_2d_new = {}
        self.trajectories_3d = {}
        self.trajectories_3d2d = {}
        self.input = [] # Used for training
        self.ground_truth = [] # Used for training

        self._fps = fps # TODO

        repeat = 0

        for idx in sorted(os.listdir(dataset_path)):
            if not os.path.isdir(os.path.join(dataset_path,idx)):
                continue
            csv_file = os.path.join(dataset_path,idx, csvfile)
            if not os.path.exists(csv_file):
                continue

            # one_trajectory = []
            # with open(csv_file, newline='') as csvfile:
            #     rows = csv.DictReader(csvfile)
            #     for i, row in enumerate(rows):
            #         if (self._fps == 30 and i % 4 == 0) or (self._fps == 60 and i % 2 == 0) or self._fps == 300:
            #             one_trajectory.append(np.array([float(row['X']),float(row['Y']),float(row['Z']),float(row['Timestamp'])]))
            # one_trajectory = np.stack(one_trajectory, axis=0)

            one_trajectory = load_points_from_csv(csv_file)
            one_trajectory = np.stack([p.toXYZT() for p in one_trajectory if p.visibility == 1], axis=0)

            # 3D Trajectory Timestamp reset to zero
            one_trajectory[:,3] -= one_trajectory[0,3]

            if smooth_3d:
                one_trajectory[:,0], one_trajectory[:,1], one_trajectory[:,2],_,_,_ = fit_3d(one_trajectory[:,0], one_trajectory[:,1], one_trajectory[:,2], N=one_trajectory.shape[0], deg=4)
                save_points_to_csv([np2Point(p,fid=fid) for fid, p in enumerate(one_trajectory)], csv_file=os.path.join(dataset_path,idx,'smooth_3d.csv'))

            self.trajectories_3d[int(idx)] = one_trajectory.copy()

            curve_2d, curve_3d2d, slope, intercept = FitVerticalPlaneTo2D(one_trajectory, smooth_2d=smooth_2d, poly=poly, smooth_2d_x_accel=True)

            # curve_2d_new, _, _, _ = FitVerticalPlaneTo2D(one_trajectory, smooth_2d=smooth_2d, poly=poly, smooth_2d_x_accel=True)
            # self.trajectories_2d_new[int(idx)] = curve_2d_new.copy()
            self.trajectories_2d[int(idx)] = curve_2d.copy()
            self.trajectories_3d2d[int(idx)] = curve_3d2d.copy()

            if network == 'blstm':
                if curve_2d.shape[0] < N+1:
                    # no ground truth
                    continue
                for i in range(curve_2d.shape[0]-N-N+1+repeat):
                    self.input.append(curve_2d[i:i+N])
                    self.ground_truth.append(curve_2d[i+N-repeat:i+N+N-repeat])
            elif network == 'seq2seq':
                if curve_2d.shape[0] < N+1:
                    # no ground truth
                    continue
                self.input.append(curve_2d[0:N])
                self.ground_truth.append(curve_2d[N:])
            elif network is None:
                pass
            else:
                print(f"Unsupported network & mode: {network}")
                sys.exit(1)
        if network == 'blstm' or network == 'seq2seq':
            self.input = np.stack(self.input, axis=0)
            self.ground_truth = np.stack(self.ground_truth, axis=0)

            for i in range(self.input.shape[0]):
                # For each training data, assume the first point's Timestamp as zero
                self.ground_truth[i,:,-1] -= self.input[i,0,-1]
                self.input[i,:,-1] -= self.input[i,0,-1]
                if move_origin_2d:
                    self.ground_truth[i,:,:-1] -= self.input[i,0,:-1]
                    self.input[i,:,:-1] -= self.input[i,0,:-1]
        if network == 'blstm':
            assert self.input.shape[0] == self.ground_truth.shape[0], "[DataLoader] Wrong!"

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        return self.input[index], self.ground_truth[index]

    def whole_2d(self):
        # {1:t1, 2:t2, 3:t3, 4:t4} ...
        # t1,t2,t3,t4 : unequal length
        # t1 : np array, shape: [?, 3]  3 represents (XY, Z, t)
        return self.trajectories_2d

    def whole_3d(self):
        # {1:t1, 2:t2, 3:t3, 4:t4} ...
        # t1,t2,t3,t4 : unequal length
        # t1 : np array, shape: [?, 4]  4 represents (X, Y, Z, t)
        return self.trajectories_3d

    def whole_3d2d(self):
        # {1:t1, 2:t2, 3:t3, 4:t4} ...
        # t1,t2,t3,t4 : unequal length
        # t1 : np array, shape: [?, 4]  4 represents (X, 0, Z, t)
        return self.trajectories_3d2d

    def fps(self):
        return self._fps

    def show_each(self):
        for idx, data in self.whole_2d().items():
            # assert data.shape[0] == self.trajectories_2d_new[idx].shape[0], "??"
            # plt.scatter(data[:,0],data[:,1],s=[10 if data.shape[0]%(i+10) == 10 else 1 for i in range(data.shape[0]) ], color='blue')
            # plt.scatter(self.trajectories_2d_new[idx][:,0],self.trajectories_2d_new[idx][:,1],s=[10 if data.shape[0]%(i+10) == 10 else 1 for i in range(data.shape[0])], color='red')
            plt.plot(data[:,0],data[:,1],marker='o',markersize=2)
            plt.title(f"idx: {idx}")
            plt.show()

class PhysicsDataSet_blstm(torch.utils.data.Dataset):
    def __init__(self, N=5, datas=0):

        self.input = [] # Used for training
        self.ground_truth = [] # Used for training

        fps_range = (25.0,150.0)
        elevation_range = (-89.0,89.0)
        speed_range = (5.0,250.0) # km/hr

        random_datas = np.random.uniform(low =[fps_range[0], elevation_range[0], speed_range[0]],
                                                high=[fps_range[-1],elevation_range[-1],speed_range[-1]],
                                                size=(datas,3))
        print("===BLSTM Physics Dataset===")
        print(f"FPS: {fps_range[0]} ~ {fps_range[-1]}")
        print(f"Elevation: {elevation_range[0]} ~ {elevation_range[-1]} degree")
        print(f"Speed: {speed_range[0]} ~ {speed_range[-1]} km/hr")
        print(f"Datas: {random_datas.shape[0]}")

        starting_point = [0, 0, 0]

        for fps,e,s in random_datas:
            in_t = np.arange(0,N)*(1/fps)
            out_t = np.arange(0,N)*(1/fps) + in_t[-1] + (1/fps)
            teval = np.concatenate((in_t, out_t))

            s = s * 1000/3600 # km/hr -> m/s
            initial_velocity = [s * math.cos(e/180*math.pi), 0, s * math.sin(e/180*math.pi)]
            traj = solve_ivp(bm_ball, [0, teval[-1]], starting_point + initial_velocity, t_eval = teval) # traj.t traj.y
            xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
            t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)

            trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

            assert trajectories.shape[0] == N*2, "[Physics Dataloader] BLSTM datas shape wrong."

            self.input.append(trajectories[0:N,[0,2,3]])
            self.ground_truth.append(trajectories[N:N*2,[0,2,3]])
        
        self.input = np.stack(self.input, axis=0)
        self.ground_truth = np.stack(self.ground_truth, axis=0)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        return self.input[index], self.ground_truth[index]

class PhysicsDataSet_seq2seq(torch.utils.data.Dataset):
    def __init__(self, datas=0, in_max_drop=0.0, in_max_time=0.2, out_max_time=4, move_origin_2d=True):
        # move_origin_2d: move the curve (self.src + self.trg) to origin

        self.trajectories_2d = {}
        self.trajectories_3d2d = {}


        self.src = [] # Used for training
        self.trg = [] # Used for training
        self.output_fps = [] # Used for training (Seq2Seq)

        fps_range = (25.0,150.0)
        # fps_range = (118.0,122.0)
        elevation_range = (-80.0,80.0)
        speed_range = (5.0,250.0) # km/hr
        output_fps_range = (25.0,150.0) # Only used for Seq2Seq
        # output_fps_range = (118.0,122.0) # Only used for Seq2Seq

        random_datas = np.random.uniform(low =[fps_range[0], elevation_range[0], speed_range[0], output_fps_range[0]],
                                                high=[fps_range[-1],elevation_range[-1],speed_range[-1],output_fps_range[-1]],
                                                size=(datas,4))
        print("===Physics Dataset===")
        print(f"FPS: {fps_range[0]} ~ {fps_range[-1]}")
        print(f"Elevation: {elevation_range[0]} ~ {elevation_range[-1]} degree")
        print(f"Speed: {speed_range[0]} ~ {speed_range[-1]} km/hr")
        print(f"Output Fps: {output_fps_range[0]} ~ {output_fps_range[-1]}")
        print(f"In Max Drop: {in_max_drop}")
        print(f"In Max Time: {in_max_time}s")
        print(f"Out Max Time: {out_max_time}s")
        print(f"Datas: {random_datas.shape[0]}")
        # print(f"Datas for each fps: {datas}. Total: {datas*(fps_range[-1]-fps_range[0]+1)}")

        starting_point = [0, 0, 3]

        idx = 1
        for fps,e,s,output_fps in random_datas:
            assert in_max_drop >= 0.0 and in_max_drop < 1.0, "in_max_drop should between 0.0~1.0"
            in_time = random.uniform(2/fps, in_max_time)
            in_drop = random.uniform(0.0, in_max_drop)
            in_t = sorted(np.random.choice(np.arange(0,in_time*fps)*(1/fps),
                    size=random.randint(2,int(in_time*fps*(1-in_drop))), # at least 2 point
                    replace=False))
            in_t = in_t - in_t[0] # reset time to zero
            out_t = np.arange(0,out_max_time*output_fps)*(1/output_fps) + in_t[-1] + (1/output_fps)

            teval = np.concatenate((in_t, out_t))

            s = s * 1000/3600 # km/hr -> m/s
            initial_velocity = [s * math.cos(e/180*math.pi), 0, s * math.sin(e/180*math.pi)]
            traj = solve_ivp(bm_ball, [0, teval[-1]], starting_point + initial_velocity, t_eval = teval) # traj.t traj.y
            xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
            t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)

            trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

            assert len(in_t)+len(out_t) == trajectories.shape[0], " "


            self.src.append(trajectories[:len(in_t),[0,2,3]])
            self.trg.append(trajectories[len(in_t):,[0,2,3]])
            self.output_fps.append(output_fps)

        
            self.trajectories_2d[int(idx)] = trajectories[:,[0,2,3]].copy()
            self.trajectories_3d2d[int(idx)] = trajectories.copy()
            idx += 1

        for i in range(len(self.src)):
            # For each training data, assume the first point's Timestamp as zero
            self.trg[i][:,-1] -= self.src[i][0,-1]
            self.src[i][:,-1] -= self.src[i][0,-1]
            if move_origin_2d:
                self.trg[i][:,:-1] -= self.src[i][0,:-1]
                self.src[i][:,:-1] -= self.src[i][0,:-1]

        # stack with zero padding
        # max_len = 0
        # for tra in self.trg:
        #     max_len = max(max_len,tra.shape[0])
        # for i in range(len(self.trg)):
        #     tmp = np.zeros((max_len, 3)) # XY, Z, t
        #     tmp[:self.trg[i].shape[0]] = self.trg[i]
        #     self.trg[i] = tmp
        # self.trg = np.stack(self.trg, axis=0)
        # pass

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.output_fps[index]

    def whole_2d(self):
        return self.trajectories_2d

    def whole_3d(self):
        return self.trajectories_3d2d

    def whole_3d2d(self):
        return self.trajectories_3d2d


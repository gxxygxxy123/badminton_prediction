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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from scipy.integrate import solve_ivp

from scipy.signal import savgol_filter
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from physic_model import bm_ball

sys.path.append(f"../lib")
from point import Point, load_points_from_csv, save_points_to_csv, np2Point



def points_change_fps(points_list: list, fps):
    points = points_list.copy()
    new_points = []
    for i in range(len(points)-1):
        assert points[i].timestamp < points[i+1].timestamp, "Points ts isn't sorted"
    init_ts = points[0].timestamp
    for i in range(len(points)):
        assert points[i].visibility == 1, "Points Vis != 1." # TO DELETE should be warning
        points[i].timestamp -= init_ts

    ts = 0.0
    fid = 0

    for i in range(len(points)-1):
        while points[i].timestamp <= ts and points[i+1].timestamp >= ts:
            x = (points[i].x * (points[i+1].timestamp - ts) + points[i+1].x * (ts-points[i].timestamp)) / (points[i+1].timestamp - points[i].timestamp)
            y = (points[i].y * (points[i+1].timestamp - ts) + points[i+1].y * (ts-points[i].timestamp)) / (points[i+1].timestamp - points[i].timestamp)
            z = (points[i].z * (points[i+1].timestamp - ts) + points[i+1].z * (ts-points[i].timestamp)) / (points[i+1].timestamp - points[i].timestamp)
            v = 1
            new_points.append(Point(fid=fid, timestamp=ts, visibility=v, x=x, y=y, z=z))
            fid += 1
            ts += 1/fps

    return new_points


class RNNDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, fps=None, N=None, smooth_2d=True, smooth_3d=False, poly=8, network=None, csvfile='Model3D.csv'):

        super(RNNDataSet).__init__()
        self.trajectories_2d = {}
        self.trajectories_3d = {}
        self.trajectories_3d2d = {}
        self.src = [] # Used for training
        self.trg = [] # Used for training

        self.N = N

        self.dataset_fps = fps
        self.dt = []

        seq2seq_dxyt = False
        seq2seq_indim = [0,1]
        print(f"RNNDataset seq2seq_dxyt: {seq2seq_dxyt}, seq2seq_indim: {seq2seq_indim}")

        debug_move_ori = True
        if debug_move_ori:
            print(f"RNNDataset debug_moveori on[Finding Bug]")

        for idx in sorted(os.listdir(dataset_path)):
            if not os.path.isdir(os.path.join(dataset_path,idx)):
                continue
            csv_file = os.path.join(dataset_path,idx, csvfile)
            if not os.path.exists(csv_file):
                continue

            # Load Data From csv
            points = load_points_from_csv(csv_file)

            # 3D Trajectory Timestamp reset to zero
            t0 = points[0].timestamp
            for p in points:
                p.timestamp -= t0

            # Calculate the dataset FPS
            if self.dataset_fps == None:
                for i in range(len(points)-1):
                    self.dt.append(points[i+1].timestamp-points[i].timestamp)
            # Change ts to fixed fps
            else:
                points = points_change_fps(points, self.dataset_fps)

            # Convert Data to numpy array
            one_trajectory = np.stack([p.toXYZT() for p in points if p.visibility == 1], axis=0)

            if smooth_3d:
                one_trajectory[:,0], one_trajectory[:,1], one_trajectory[:,2],_,_,_ = fit_3d(one_trajectory[:,0], one_trajectory[:,1], one_trajectory[:,2], N=one_trajectory.shape[0], deg=4)
                save_points_to_csv([np2Point(p,fid=fid) for fid, p in enumerate(one_trajectory)], csv_file=os.path.join(dataset_path,idx,'smooth_3d.csv'))

            self.trajectories_3d[int(idx)] = one_trajectory.copy()

            curve_2d, curve_3d2d, slope, intercept = FitVerticalPlaneTo2D(one_trajectory, smooth_2d=smooth_2d, poly=poly, smooth_2d_x_accel=True)
            #curve_2d = [M,3]
            #curve_3d2d = [M,4]

            assert curve_2d[0,2] == 0.0 and curve_3d2d[0,3] == 0.0

            # debug
            if debug_move_ori:
                curve_2d[:,[0,1]] -= curve_2d[0,[0,1]]
                curve_3d2d[:,[0,1,2]] -= curve_3d2d[0,[0,1,2]]

            self.trajectories_2d[int(idx)] = curve_2d.copy()
            self.trajectories_3d2d[int(idx)] = curve_3d2d.copy()

            if network == 'blstm':
                if curve_2d.shape[0] < N+1:
                    # no ground truth
                    continue
                for i in range(curve_2d.shape[0]-N-N+1):
                    self.src.append(curve_2d[i:i+N])
                    self.trg.append(curve_2d[i+N:i+N+N])
            elif network == 'seq2seq':
                pass
                # if curve_2d.shape[0] < N+1:
                #     # no ground truth
                #     continue
                # # dx,dy,dt
                # if seq2seq_dxyt:
                #     self.src.append(torch.from_numpy(np.diff(curve_2d[:N,seq2seq_indim],axis=0)))
                #     self.trg.append(torch.from_numpy(np.diff(curve_2d[N-1:,seq2seq_indim],axis=0)))
                # # x,y,t
                # else:
                #     self.src.append(torch.from_numpy(curve_2d[:N,seq2seq_indim]-curve_2d[0,seq2seq_indim]))
                #     self.trg.append(torch.from_numpy(curve_2d[N:,seq2seq_indim]-curve_2d[0,seq2seq_indim]))
            elif network == 'transformer':
                if curve_2d.shape[0] < N+1:
                    # no ground truth
                    continue
                self.src.append(torch.from_numpy(np.diff(curve_2d[:N],axis=0)))
                self.trg.append(torch.from_numpy(np.diff(curve_2d[N-1:],axis=0)))
            elif network is None:
                pass
            else:
                print(f"Unsupported network & mode: {network}")
                sys.exit(1)
        # if network == 'blstm' or network == 'seq2seq':
        #     self.src = np.stack(self.src, axis=0)
        #     if network == 'blstm':
        #         self.trg = np.stack(self.trg, axis=0)

        # if network == 'blstm':
        #     assert self.src.shape[0] == self.trg.shape[0], "[DataLoader] Wrong!"

        if self.dataset_fps == None:
            self.dataset_fps = 1/(sum(self.dt)/len(self.dt))

        if network == 'transformer':
            max_src_length = 100
            for i in range(len(self.src)):
                self.src[i] = nn.functional.pad(self.src[i],(0,0,0,max_src_length-self.src[i].size(-2)),'constant')
            max_trg_length = 1000
            for i in range(len(self.trg)):
                self.trg[i] = nn.functional.pad(self.trg[i],(0,0,0,max_trg_length-self.trg[i].size(-2)),'constant')

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index], torch.tensor(self.fps()).float() # TODO

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
        return self.dataset_fps

    def show_each(self):
        for idx, data in self.whole_2d().items():
            # assert data.shape[0] == self.trajectories_2d_new[idx].shape[0], "??"
            # plt.scatter(data[:,0],data[:,1],s=[10 if data.shape[0]%(i+10) == 10 else 1 for i in range(data.shape[0]) ], color='blue')
            # plt.scatter(self.trajectories_2d_new[idx][:,0],self.trajectories_2d_new[idx][:,1],s=[10 if data.shape[0]%(i+10) == 10 else 1 for i in range(data.shape[0])], color='red')
            plt.plot(data[:,0],data[:,1],marker='o',markersize=2)
            plt.title(f"idx: {idx}")
            plt.show()



class PhysicsDataSet(torch.utils.data.Dataset):
    def __init__(self, datas=0, in_max_time=0.2, out_max_time=2.0, model='seq2seq'):

        # Option
        cut_under_ground = False # It makes forecasting bad
        noise_t = True
        noise_xy = True
        dxyt = False
        network_in_dim = 3 # 2: x,y 3: x,y,t
        drop_mode = 0 # 0: fixed, 1: unequal length but continue, 2: random drop
        starting_point = [0, 0, 3]

        self.trajectories_2d = {}
        self.trajectories_3d2d = {}

        self.src = [] # Used for training
        self.trg = [] # Used for training
        if model == 'transformer':
            self.trg_in = [] # Used for decoder input

        self.output_fps = [] # Used for training

        self.model = model

        fps_range = (120.0,120.0)
        elevation_range = (-40.0,60.0)
        speed_range = (30.0,200.0) # km/hr
        output_fps_range = (120.0,120.0) # Only used for Seq2Seq

        random_datas = np.random.uniform(low =[fps_range[0], elevation_range[0], speed_range[0], output_fps_range[0]],
                                                high=[fps_range[-1],elevation_range[-1],speed_range[-1],output_fps_range[-1]],
                                                size=(datas,4))

        debug_move_ori = True
        if debug_move_ori:
            starting_point = [0,0,0]
            print(f"PhysicsDataset debug_moveori on[Finding Bug]")


        print(f"===Physics Dataset ({model})===\n"
              f"FPS: {fps_range[0]} ~ {fps_range[-1]}\n"
              f"Elevation: {elevation_range[0]} ~ {elevation_range[-1]} degree\n"
              f"Speed: {speed_range[0]} ~ {speed_range[-1]} km/hr\n"
              f"Output Fps: {output_fps_range[0]} ~ {output_fps_range[-1]}\n"
              f"In Max Time: {in_max_time}s\n"
              f"Out Max Time: {out_max_time}s\n"
              f"Datas: {random_datas.shape[0]}\n"
              f"========Physics Dataset Option========\n"
              f"Cut Under ground: {cut_under_ground}\n"
              f"Noise t: {noise_t}\n"
              f"Noise xy: {noise_xy}\n"
              f"dxyt: {dxyt}\n"
              f"network input dim: {network_in_dim}\n"
              f"drop point mode: {drop_mode}\n"
              f"starting point: {starting_point}\n")

        idx = 1
        for fps,e,s,output_fps in random_datas:
            if drop_mode == 2:
                in_t = np.sort(np.random.choice(np.arange(0,in_max_time*fps)*(1/fps),
                        size=round(random.uniform(2,in_max_time*fps)), # at least 1 vector
                        replace=False))
            elif drop_mode == 1:
                in_t = np.arange(0,round(random.uniform(2,in_max_time*fps)))*(1/fps)
            elif drop_mode == 0:
                in_t = np.arange(0,in_max_time*fps)*(1/fps)

            out_t = np.arange(0,out_max_time*output_fps)*(1/output_fps) + in_t[-1] + (1/output_fps)

            teval = np.concatenate((in_t, out_t))

            # Add noise to dt
            if noise_t:
                teval += np.random.normal(0,0.02/fps, teval.shape) # 1/fps / 50

            # reset time to zero
            teval = teval - teval[0]

            s = s * 1000/3600 # km/hr -> m/s
            initial_velocity = [s * math.cos(e/180*math.pi), 0, s * math.sin(e/180*math.pi)]
            traj = solve_ivp(bm_ball, [0, teval[-1]], starting_point + initial_velocity, t_eval = teval) # traj.t traj.y
            xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
            t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)

            trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

            assert len(in_t)+len(out_t) == trajectories.shape[0], " "

            # # Cut under ground part
            if cut_under_ground:
                while(trajectories[-1][2] <= 0):
                    trajectories = trajectories[:-1] # pop last point
                if trajectories.shape[0] <= len(in_t):
                    continue

            # Add noise to dx,dy (except starting point)
            if noise_xy:
                trajectories[1:,[0,2]] += np.random.normal(0,0.02, trajectories[1:,[0,2]].shape) # 2cm

            if network_in_dim == 2:
                in_dim = [0,2] # x,y (in 2d)
            elif network_in_dim == 3:
                in_dim = [0,2,3] # x,y,t (in 2d)

            # Input (dx,dy)/(dx,dy,dt)
            if dxyt:
                self.src.append(torch.from_numpy(np.diff(trajectories[:len(in_t),in_dim],axis=0)))
                self.trg.append(torch.from_numpy(np.diff(trajectories[len(in_t)-1:,in_dim],axis=0)))
                if model == 'transformer':
                    self.trg_in.append(torch.from_numpy(np.diff(trajectories[len(in_t)-1-1:-1,in_dim],axis=0)))
            # Input (x,y)/(x,y,t)
            else:
                trajectories[:,[0,2]] -= trajectories[0,[0,2]] # move x,y to origin
                assert not np.any(trajectories[0]), "x,y,t is not 0"

                ### debug output at (0,0)*
                #trajectories[:,[0,2]] -= trajectories[len(in_t)-1,[0,2]]

                self.src.append(torch.from_numpy(trajectories[:len(in_t),in_dim]))
                self.trg.append(torch.from_numpy(trajectories[len(in_t):,in_dim]))
                if model == 'transformer':
                    self.trg_in.append(torch.from_numpy(trajectories[len(in_t)-1:-1,in_dim]))

            self.output_fps.append(torch.tensor(output_fps).float())

            self.trajectories_2d[int(idx)] = trajectories[:,[0,2,3]].copy()
            self.trajectories_3d2d[int(idx)] = trajectories.copy()
            idx += 1

        if model == 'seq2seq':
            print(f"src mean length: {sum([i.shape[0] for i in self.src])/len(self.src)}")
            print(f"src max length: {max([i.shape[0] for i in self.src])}")
            print(f"src min length: {min([i.shape[0] for i in self.src])}")
            # print(f"Debug padding {max([i.shape[0] for i in self.src])}!!!!!!!!!!!!!!!!!!!!")
            # Shouldn't be added. 07/04
            # for i in range(len(self.src)): # debug
            #     self.src[i] = nn.functional.pad(self.src[i],(0,0,0,max([i.shape[0] for i in self.src])-self.src[i].size(-2)),'constant')

        if model == 'transformer':
            max_src_length=100
            max_trg_length=1000
            for i in range(len(self.src)):
                self.src[i] = nn.functional.pad(self.src[i],(0,0,0,max_src_length-self.src[i].size(-2)),'constant')
            for i in range(len(self.trg)):
                self.trg[i] = nn.functional.pad(self.trg[i],(0,0,0,max_trg_length-self.trg[i].size(-2)),'constant')
            for i in range(len(self.trg_in)):
                self.trg_in[i] = nn.functional.pad(self.trg_in[i],(0,0,0,max_trg_length-self.trg_in[i].size(-2)),'constant')

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        if self.model == 'transformer':
            return self.src[index], self.trg_in[index], self.trg[index]
        return self.src[index], self.trg[index], self.output_fps[index]

    def whole_2d(self):
        return self.trajectories_2d

    def whole_3d(self):
        return self.trajectories_3d2d

    def whole_3d2d(self):
        return self.trajectories_3d2d

    def src_maxlen(self):
        return max(tra.size(dim=0) for tra in self.src)
    def trg_maxlen(self):
        return max(tra.size(dim=0) for tra in self.trg)

'''
class PhysicsDataSet_blstm(torch.utils.data.Dataset):
    def __init__(self, N=5, datas=0):

        self.src = [] # Used for training
        self.trg = [] # Used for training

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

            self.src.append(trajectories[0:N,[0,2,3]])
            self.trg.append(trajectories[N:N*2,[0,2,3]])

        self.src = np.stack(self.src, axis=0)
        self.trg = np.stack(self.trg, axis=0)

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        return self.src[index], self.trg[index]
'''



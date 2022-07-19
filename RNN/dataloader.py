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

        #seq2seq_dxyt = False
        #seq2seq_indim = [0,1]
        #print(f"RNNDataset seq2seq_dxyt: {seq2seq_dxyt}, seq2seq_indim: {seq2seq_indim}")

        debug_move_ori = False
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
            elif network == 'TF':
                pass
                # if curve_2d.shape[0] < N+1:
                #     # no ground truth
                #     continue
                # self.src.append(torch.from_numpy(np.diff(curve_2d[:N],axis=0)))
                # self.trg.append(torch.from_numpy(np.diff(curve_2d[N-1:],axis=0)))
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

        if network == 'TF':
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
            plt.plot(data[:,0],data[:,1],marker='o',markersize=2)
            plt.title(f"idx: {idx}")
            plt.show()


class PhysicsDataSet(torch.utils.data.Dataset):
    def __init__(self, datas=0,
                       in_max_time=0.2,
                       out_max_time=3.0,
                       cut_under_ground=False,
                       noise_t=False,
                       noise_xy=True,
                       dxyt=True,
                       network_in_dim=2, # 2: x,y 3: x,y,t
                       drop_mode=0, # 0: fixed, 1: unequal length but continue, 2: random drop
                       fps_range = (120.0,120.0),
                       elevation_range = (-80.0,80.0),
                       speed_range = (30.0,200.0), # km/hr
                       output_fps_range = (120.0,120.0),
                       starting_point=[0, 0, 3],
                       model='TF'):

        self.data = {}

        data_src = []
        data_trg = []

        self.model = model

        random_datas = np.random.uniform(low =[fps_range[0], elevation_range[0], speed_range[0], output_fps_range[0]],
                                                high=[fps_range[-1],elevation_range[-1],speed_range[-1],output_fps_range[-1]],
                                                size=(datas,4))

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
                in_dim = np.array([0,2]) # x,y (in 2d)
            elif network_in_dim == 3:
                in_dim = np.array([0,2,3]) # x,y,t (in 2d)

            # Input (dx,dy)/(dx,dy,dt)
            if dxyt:
                #tmp = np.concatenate((np.zeros((1,in_dim.shape[0])), np.diff(trajectories[:,in_dim],axis=0)),0)
                tmp = np.diff(trajectories[:,in_dim],axis=0)
                data_src.append(tmp[:len(in_t)])
                data_trg.append(tmp[len(in_t):])
            # Input (x,y)/(x,y,t)
            else:
                tmp = trajectories[:,in_dim] - trajectories[0,in_dim] # move x,y,t to origin
                assert not np.any(tmp[0]), "x,y,t is not 0"
                data_src.append(tmp[:len(in_t)])
                data_trg.append(tmp[len(in_t):])

            idx += 1
        
        # TODO TF train with different seq length?
        if self.model == 'TF':
            self.data['src'] = np.stack(data_src,0)
            self.data['trg'] = np.stack(data_trg,0)

        elif self.model == 'BLSTM' or self.model == 'Seq2Seq':
            data_src = sorted(data_src, key=lambda x: len(x), reverse=True)
            src_lens = [len(x) for x in data_src]
            for i in range(len(data_src)):
                data_src[i] = np.pad(data_src[i],((0,max(src_lens)-len(data_src[i])),(0,0)), 'constant', constant_values=0)

            data_trg = sorted(data_trg, key=lambda x: len(x), reverse=True)
            trg_lens = [len(x) for x in data_trg]
            for i in range(len(data_trg)):
                data_trg[i] = np.pad(data_trg[i],((0,max(trg_lens)-len(data_trg[i])),(0,0)), 'constant', constant_values=0)

            self.data['src'] = np.stack(data_src,0)
            self.data['trg'] = np.stack(data_trg,0)
            # (datas, max src/trg len, features)

            self.data['src_lens'] = src_lens
            self.data['trg_lens'] = trg_lens

        self._mean = np.nanmean(np.where(self.data['src']!=0,self.data['src'],np.nan),(0,1))
        self._std = np.nanstd(np.where(self.data['src']!=0,self.data['src'],np.nan),(0,1))

    def __len__(self):
        if self.model == 'TF':
            return self.data['src'].shape[0]
        elif self.model == 'BLSTM' or self.model == 'Seq2Seq':
            return len(self.data['src'])

    def __getitem__(self, index):
        if self.model == 'TF':
            return {'src':torch.Tensor(self.data['src'][index]), 'trg':torch.Tensor(self.data['trg'][index])}
        elif self.model == 'BLSTM' or self.model == 'Seq2Seq':
            return {'src':torch.Tensor(self.data['src'][index]), 'trg':torch.Tensor(self.data['trg'][index]),
                    'src_lens':self.data['src_lens'][index], 'trg_lens':self.data['trg_lens'][index]}

    def mean(self):
        return torch.Tensor(self._mean)

    def std(self):
        return torch.Tensor(self._std)

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
    def __init__(self, dataset_path: str, fps=None, N=10, move_origin_2d=True, smooth_2d=True, smooth_3d=False, poly=8, network=None, csvfile='Model3D.csv'):

        # move_origin_2d: move the curve (self.src + self.trg) to origin
        super(RNNDataSet).__init__()
        self.trajectories_2d = {}
        self.trajectories_3d = {}
        self.trajectories_3d2d = {}
        self.src = [] # Used for training
        self.trg = [] # Used for training

        self.N = N

        self.dataset_fps = fps
        self.dt = []

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
                if curve_2d.shape[0] < N+1:
                    # no ground truth
                    continue
                self.src.append(torch.from_numpy(np.diff(curve_2d[:N,[0,1]],axis=0))) # debug remove t
                self.trg.append(torch.from_numpy(np.diff(curve_2d[N-1:,[0,1]],axis=0))) # debug remove t
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

    def N(self):
        return self.N

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


class PhysicsDataSet_seq2seq(torch.utils.data.Dataset):
    def __init__(self, datas=0, in_max_time=0.2, out_max_time=0.2):

        self.trajectories_2d = {}
        self.trajectories_3d2d = {}


        self.src = [] # Used for training
        self.trg = [] # Used for training
        self.output_fps = [] # Used for training

        fps_range = (25.0,150.0)
        elevation_range = (-80.0,80.0)
        speed_range = (5.0,250.0) # km/hr
        output_fps_range = (25.0,150.0) # Only used for Seq2Seq

        random_datas = np.random.uniform(low =[fps_range[0], elevation_range[0], speed_range[0], output_fps_range[0]],
                                                high=[fps_range[-1],elevation_range[-1],speed_range[-1],output_fps_range[-1]],
                                                size=(datas,4))

        print("===Physics Dataset===")
        print(f"FPS: {fps_range[0]} ~ {fps_range[-1]}")
        print(f"Elevation: {elevation_range[0]} ~ {elevation_range[-1]} degree")
        print(f"Speed: {speed_range[0]} ~ {speed_range[-1]} km/hr")
        print(f"Output Fps: {output_fps_range[0]} ~ {output_fps_range[-1]}")

        print(f"In Max Time: {in_max_time}s")
        print(f"Out Max Time: {out_max_time}s")
        print(f"Datas: {random_datas.shape[0]}")

        starting_point = [0, 0, 3]

        debug = True
        if debug:
            debug_fps = 60
            self.debug_fps = debug_fps
            print(f"Debug: {debug} out fps: {debug_fps}, Cut ground, No Add noise in in_t !!!!!!!!!!!!!!")

        idx = 1
        for fps,e,s,output_fps in random_datas:
            # Debug
            if debug:
                fps = debug_fps
                output_fps = debug_fps

            # in_t = np.sort(np.random.choice(np.arange(0,in_max_time*fps)*(1/fps),
            #         size=round(random.uniform(2,in_max_time*fps)), # at least 1 vector
            #         replace=False))
            in_t = np.arange(0,round(random.uniform(2,in_max_time*fps)))*(1/fps)

            # Add noise to in_t
            # in_t += np.random.normal(0,0.1/fps,in_t.shape)
            # in_t = in_t - in_t[0] # reset time to zero

            out_t = np.arange(0,out_max_time*output_fps)*(1/output_fps) + in_t[-1] + (1/output_fps)

            teval = np.concatenate((in_t, out_t))

            s = s * 1000/3600 # km/hr -> m/s
            initial_velocity = [s * math.cos(e/180*math.pi), 0, s * math.sin(e/180*math.pi)]
            traj = solve_ivp(bm_ball, [0, teval[-1]], starting_point + initial_velocity, t_eval = teval) # traj.t traj.y
            xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
            t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)

            trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

            assert len(in_t)+len(out_t) == trajectories.shape[0], " "

            # Cut under ground part
            while(trajectories[-1][2] <= 0):
                trajectories = trajectories[:-1] # pop last point
            if trajectories.shape[0] <= len(in_t):
                continue

            # Input (x,y) is vector
            self.src.append(torch.from_numpy(np.diff(trajectories[:len(in_t),[0,2]],axis=0))) # debug remove t
            self.trg.append(torch.from_numpy(np.diff(trajectories[len(in_t)-1:,[0,2]],axis=0)))
            self.output_fps.append(torch.tensor(output_fps).float())

            self.trajectories_2d[int(idx)] = trajectories[:,[0,2,3]].copy()
            self.trajectories_3d2d[int(idx)] = trajectories.copy()
            idx += 1

        """ Input (x,y) not vector

        for i in range(len(self.src)):
            # For each training data, assume the first point's Timestamp as zero
            self.trg[i][:,-1] -= self.src[i][0,-1]
            self.src[i][:,-1] -= self.src[i][0,-1]

            # Move first point(x,y) to (0,0)
            self.trg[i][:,:-1] -= self.src[i][0,:-1]
            self.src[i][:,:-1] -= self.src[i][0,:-1]
        """
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

        print(max([i.shape[0] for i in self.src]))

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

    def fps(self):
        return self.debug_fps #DEBUG

    def src_maxlen(self):
        return max(tra.size(dim=0) for tra in self.src)
    def trg_maxlen(self):
        return max(tra.size(dim=0) for tra in self.trg)

class PhysicsDataSet_transformer(torch.utils.data.Dataset):
    def __init__(self, datas=0, output_fps=300, in_max_time=0.2, out_max_time=4, max_src_length=100, max_trg_length=1000):

        self.trajectories_2d = {}
        self.trajectories_3d2d = {}


        self.src = [] # Used for training
        self.trg = [] # Used for decoder input
        self.trg_y = [] # Used for loss
        self.output_fps = output_fps

        fps_range = (25.0,150.0)
        # fps_range = (118.0,122.0)
        elevation_range = (-80.0,80.0)
        speed_range = (5.0,250.0) # km/hr


        random_datas = np.random.uniform(low =[fps_range[0], elevation_range[0], speed_range[0]],
                                                high=[fps_range[-1],elevation_range[-1],speed_range[-1]],
                                                size=(datas,3))

        print("===Physics Dataset===")
        print(f"FPS: {fps_range[0]} ~ {fps_range[-1]}")
        print(f"Elevation: {elevation_range[0]} ~ {elevation_range[-1]} degree")
        print(f"Speed: {speed_range[0]} ~ {speed_range[-1]} km/hr")
        print(f"Output Fps: {self.output_fps}")

        print(f"In Max Time: {in_max_time}s")
        print(f"Out Max Time: {out_max_time}s")
        print(f"Datas: {random_datas.shape[0]}")

        starting_point = [0, 0, 3]

        idx = 1
        for fps,e,s in random_datas:

            # in_t = sorted(np.random.choice(np.arange(0,in_max_time*fps)*(1/fps),
            #         size=round(random.uniform(2,in_max_time*fps)), # at least 2 point
            #         replace=False))
            #in_t = in_t - in_t[0] # reset time to zero

            in_t = np.arange(0,round(random.uniform(2,in_max_time*fps)))*(1/fps)

            out_t = np.arange(0,out_max_time*self.output_fps)*(1/self.output_fps) + in_t[-1] + (1/self.output_fps)

            teval = np.concatenate((in_t, out_t))

            s = s * 1000/3600 # km/hr -> m/s
            initial_velocity = [s * math.cos(e/180*math.pi), 0, s * math.sin(e/180*math.pi)]
            traj = solve_ivp(bm_ball, [0, teval[-1]], starting_point + initial_velocity, t_eval = teval) # traj.t traj.y
            xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
            t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)

            trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

            assert len(in_t)+len(out_t) == trajectories.shape[0], " "

            # Input (x,y) is vector
            self.src.append(torch.from_numpy(np.diff(trajectories[:len(in_t),[0,2,3]],axis=0)))
            self.trg.append(torch.from_numpy(np.diff(trajectories[len(in_t)-1-1:-1,[0,2,3]],axis=0)))
            self.trg_y.append(torch.from_numpy(np.diff(trajectories[len(in_t)-1:,[0,2,3]],axis=0)))

            self.trajectories_2d[int(idx)] = trajectories[:,[0,2,3]].copy()
            self.trajectories_3d2d[int(idx)] = trajectories.copy()
            idx += 1

        # stack with zero padding
        for i in range(len(self.src)):
            self.src[i] = nn.functional.pad(self.src[i],(0,0,0,max_src_length-self.src[i].size(-2)),'constant')
        for i in range(len(self.trg)):
            self.trg[i] = nn.functional.pad(self.trg[i],(0,0,0,max_trg_length-self.trg[i].size(-2)),'constant')
        for i in range(len(self.trg_y)):
            self.trg_y[i] = nn.functional.pad(self.trg_y[i],(0,0,0,max_trg_length-self.trg_y[i].size(-2)),'constant')

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.trg_y[index]

    def whole_2d(self):
        return self.trajectories_2d

    def whole_3d(self):
        return self.trajectories_3d2d

    def whole_3d2d(self):
        return self.trajectories_3d2d

    def output_fps(self):
        return self.output_fps




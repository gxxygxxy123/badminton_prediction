import math
import numpy as np
import torch
import logging
import seaborn as sns
import os
import sys
import csv
import matplotlib.pyplot as plt
import time

from blstm import Blstm
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from writer import CSVWriter
from point import Point

def predict2d(curve, model, model_type, max_predict_number=300, touch_ground_stop=True, seq2seq_output_fps=None, move_origin_2d=True, repeat=0):
    # curve shape: (N,3), 3:XY,Z,t
    N = curve.shape[0]
    BATCH_SIZE = 1
    curve =np.expand_dims(curve, axis=0) # One Batch
    curve = torch.tensor(curve,dtype=torch.float)
    # curve shape: (BATCH_SIZE, TIME_SEQ_LEN, IN_SIZE)

    assert model_type == 'blstm' or model_type == 'seq2seq', "model_type should be blstm or seq2seq."

    # move to origin, reset time
    init_state = curve[:,0].clone()
    curve -= init_state

    if model_type == 'blstm':
        for i in range(max_predict_number//N): # As long as you want

            prev = curve[:,-N:,:]

            if move_origin_2d:
                tmp = prev[:,0,:]
                prev = prev - prev[:,0,:]

            pred = model(prev)
            # pred shape: (BATCH_SIZE, TIME_SEQ_LEN, IN_SIZE)
            if move_origin_2d:
                pred = pred + tmp

            curve = torch.cat((curve,pred[:,repeat:,:]),dim=1)

            if touch_ground_stop:
                if curve[0][-1][1]+init_state[0][1] <= 0: # Touch Ground
                    while(curve[0][-2][1]+init_state[0][1] <= 0):
                        curve = curve[:,:-1,:] # pop last point
                    break
        

    elif model_type == 'seq2seq':
        trg = np.zeros((BATCH_SIZE,max_predict_number,3))
        # np.set_printoptions(suppress=True)
        # print(curve)
        output = model(curve, trg, seq2seq_output_fps,0) # turn of teacher forcing

        # sys.exit(1)
        curve = torch.cat((curve,output[:,repeat:,:]),dim=1)
        if touch_ground_stop:
            for i in range(curve.shape[1]):
                if curve[0][i][1]+init_state[0][1] <= 0:
                    curve = curve[:,:i,:]
                    break

    curve += init_state
    curve = np.squeeze(curve.detach().cpu().numpy(), axis=0)

    return curve # shape: (?,3), 3:XY,Z,t, include N points



def predict3d(curve, model, model_type, max_predict_number=300, touch_ground_stop=True, seq2seq_output_fps=None):
    # curve shape: (N,4), 4:X,Y,Z,t, N default should be 5
    N = curve.shape[0]

    # 3D Trajectory Timestamp reset to zero
    curve[:,3] -= curve[0,3]

    curve_2d, curve_3d, slope, intercept = FitVerticalPlaneTo2D(curve)

    #######################################################################################################


    curve_2d_output = predict2d(curve_2d, model, model_type, max_predict_number, touch_ground_stop, seq2seq_output_fps)

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(1,1,1)

    output = np.zeros((curve_2d_output.shape[0],4))

    # ax2.plot(output[:,0],output[:,1], marker='o',markersize=4,color='red')
    # ax2.plot(output[:,0],output[:,1], marker='o',markersize=2, color='yellow', linestyle='--', dashes=(5, 10))
    # plt.show()
    #######################################################################################################
    # Rotate and Translate Trajectory(on XZ Plane) to Original Direction According to the N first Points
    # Rotation Matrix = [[costheta, -sintheta],[sintheta, costheta]]
    # theta is counter-wise direction

    costheta = 1/math.sqrt(slope*slope+1)
    sintheta = abs(slope)/math.sqrt(slope*slope+1)
    if(curve[N-1][0] - curve[0][0] < 0): # p[4].x < p[0].x opposite vector direction
        costheta = -1 * costheta
    if(curve[N-1][1] - curve[0][1] < 0): # p[4].y < p[0].y opposite vector direction
        sintheta = -1 * sintheta

    for idx in range(curve_2d_output.shape[0]):
        # Rotation
        rotated_x = costheta*curve_2d_output[idx][0] - sintheta*0.0
        rotated_y = sintheta*curve_2d_output[idx][0] + costheta*0.0

        # Translation
        output[idx][0] = rotated_x+curve[0][0]
        output[idx][1] = rotated_y+curve[0][1]
        output[idx][2] = curve_2d_output[idx][1]
        output[idx][3] = curve_2d_output[idx][2]
    #######################################################################################################

    return output # shape: (?,4), 4:X,Y,Z,t, include N points

if __name__ == '__main__':

    sns.set()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    IN_CSV = sys.argv[1]

    OUT_CSV = os.path.splitext(IN_CSV)[0] + '_predict.csv'

    # for root, dirs, files in os.walk(dataset):
    #     for file in files:
    #         if file.endswith("Model3D.csv"):
    #             print(os.path.join(root, file) )

    N = 5 # Time Sequence Number

    WEIGHT_NAME = './blstm_weight'

    model = Blstm()

    model.load_state_dict(torch.load(WEIGHT_NAME))

    model.eval()

    one_trajectory = []
    with open(IN_CSV, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            one_trajectory.append(np.array([float(row['X']),float(row['Y']),float(row['Z'])]))
    assert len(one_trajectory) >= N, "Too Short"

    one_trajectory = np.stack(one_trajectory, axis=0)

    with torch.no_grad():
        output = predict3d(one_trajectory[:N],model)

        writer = CSVWriter('name', OUT_CSV)

        for i in range(output.shape[0]):
            p = Point(x=output[i][0],y=output[i][1],z=output[i][2])
            writer.writePoints(p)

        writer.close()
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(1,1,1)
    # for idx in range(len(whole_dataset)):
    #     color = next(ax2._get_lines.prop_cycler)['color']
    #     ax2.plot(whole_dataset[idx][:,0],whole_dataset[idx][:,1], marker='o',markersize=4,color=color)
    #     ax2.plot(predict_output[idx][:,0],predict_output[idx][:,1], marker='o',markersize=2, color=color, linestyle='--', dashes=(5, 10))



    # ax2.set_xlim(-1,14)
    # ax2.set_ylim(-1,4)

    # plt.show()
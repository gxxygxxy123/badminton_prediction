import math
import numpy as np
import torch
import torch.nn as nn
import logging
import seaborn as sns
import os
import sys
import csv
import matplotlib.pyplot as plt
import time

from blstm import Blstm
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from transformer.batch import subsequent_mask

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from writer import CSVWriter
from point import Point

def predict2d(curve, model, model_type, input_fps, output_fps, output_time, touch_ground_stop=True, device=torch.device('cpu')):
    # curve shape: (N,3), 3:XY,Z,t
    assert curve.shape[-1] == 3, "curve shape should be (...,3)"
    assert curve[0,2] == 0.0

    model.eval()

    N = curve.shape[0]
    BATCH_SIZE = 1

    curve = torch.tensor(curve,dtype=torch.float).to(device)
    # curve shape: (TIME_SEQ_LEN, in dim)

    assert model_type == 'seq2seq' or model_type == 'TF', "model_type should be blstm/seq2seq/TF. bsltm TODO"

    if model_type == 'blstm': # TODO
        # move to origin, reset time
        curve -= init_state

        for i in range(max_predict_number//N): # As long as you want

            prev = curve[:,-N:,:]


            tmp = prev[:,0,:]
            prev = prev - prev[:,0,:]

            pred = model(prev)
            # pred shape: (BATCH_SIZE, TIME_SEQ_LEN, in dim)

            pred = pred + tmp

            curve = torch.cat((curve,pred[:,:,:]),dim=1)

        
        curve += init_state

    elif model_type == 'seq2seq':

        dxyt = False

        in_dim = [0,1] # [0,1]: xy [0,1,2]: xyt

        # Input (dx,dy,dt)
        if dxyt:
            src = torch.diff(curve[:,in_dim],axis=0).to(device)
        else:
        # Input (x,y,t)
            src = (curve[:,in_dim] - curve[0,in_dim]).clone().detach()
            assert not torch.any(src[0])

        #src = [src len, in dim]

        trg = [src[-1]] # list

        ## DEBUG
        #trg = [torch.zeros((model.decoder.input_dim)).to(device)]

        src_lens = src.shape[0]

        mask = model.create_mask(src.unsqueeze(0)) # BATCH_SIZE=1

        # src = nn.functional.pad(src,(0,0,0,23-src.size(-2)),'constant') # debug

        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src.unsqueeze(0), torch.LongTensor([src_lens])) # BATCH_SIZE=1

        max_len = int(output_fps*output_time)

        attentions = torch.zeros(BATCH_SIZE, max_len, src_lens).to(device)

        for i in range(max_len):
            trg_tensor = trg[-1].unsqueeze(0) # BATCH_SIZE=1

            with torch.no_grad():
                output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
                #output = [batch size, out dim]
                #attentions = [batch size, src len]
            attentions[:,i,:] = attention

            pred = output.squeeze(0)
            #pred = [out dim]

            trg.append(pred)

        # Remove first element(src[-1])
        output = torch.stack(trg[1:])

        #output = [max len, out dim]


        # output = model(src, [src_lens], trg, [trg.shape[1]], output_fps, 0) # turn of teacher forcing
        # output=[BATCH_SIZE, TIME_SEQ_LEN, in dim]

        if in_dim == [0,1]:
        # Add dt if input_dim only dx,dy
            if dxyt:
                output = nn.functional.pad(output,(0,1),'constant', 1/output_fps)
            else:
                ss = output.shape[0]
                last_t = curve[-1][2].detach().clone().cpu()
                pad_t = (torch.arange(ss)*(1/output_fps) + last_t + 1/output_fps).unsqueeze(1).to(device)
                output = torch.cat((output,pad_t),dim=1)

        # Cumsum dx,dy,dt
        if dxyt:
            output = torch.cumsum(output,dim=0) + curve[-1]
        else:
            output = output + curve[0]

        curve = torch.cat((curve, output),dim=0)

        if touch_ground_stop:
            for i in range(curve.shape[0]):
                if curve[i][1] <= 0:
                    curve = curve[:i,:]
                    break
    elif model_type == 'TF':

        if touch_ground_stop:
            for i in range(curve.shape[0]):
                if curve[i][1] <= 0:
                    curve = curve[:i,:]
                    break

    curve = curve.detach().cpu().numpy()

    assert curve.ndim == 2 and curve.shape[1] == 3

    return curve # shape: (?,3), 3:XY,Z,t, include N points

def predict2d_Seq2Seq(curve :np.array,
                      model,
                      mean: torch.Tensor,
                      std: torch.Tensor,
                      out_time: float,
                      fps: float,
                      touch_ground_stop=True,
                      device=torch.device('cpu')):
    # curve shape: (N,3), 3:XY,Z,t, pad: 0
    assert curve.ndim == 2 and curve.shape[1] == 3

    model.eval()

    curve_tensor = torch.tensor(curve,dtype=torch.float).to(device)

    src = (torch.diff(curve_tensor[:,[0,1]],dim=0) - mean.to(device)) / std.to(device)

    trg = [src[-1]] # list

    src_lens = src.shape[0]

    mask = model.create_mask(src.unsqueeze(0)) # BATCH_SIZE=1

    # src = nn.functional.pad(src,(0,0,0,23-src.size(-2)),'constant') # debug

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src.unsqueeze(0), torch.LongTensor([src_lens])) # BATCH_SIZE=1

    max_len = int(fps*out_time)

    attentions = torch.zeros(1, max_len, src_lens).to(device)

    for i in range(max_len):
        trg_tensor = trg[-1].unsqueeze(0) # BATCH_SIZE=1

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
            #output = [batch size, out dim]
            #attentions = [batch size, src len]
        attentions[:,i,:] = attention

        pred = output.squeeze(0)
        #pred = [out dim]

        trg.append(pred)

    trg = torch.stack(trg,0)

    pr = ((trg[1:,[0,1]]*std.to(device)+mean.to(device)).cumsum(0) + curve_tensor[-1,[0,1]]).cpu().numpy()

    pad_t = np.expand_dims(np.arange(0,pr.shape[0]) * (1/fps) + curve[-1,2] + 1/fps, axis=1)

    pr = np.concatenate((pr,pad_t),axis=1)

    output = np.concatenate((curve, pr), axis=0)

    if touch_ground_stop:
        for i in range(output.shape[0]):
            if output[i][1] <= 0:
                output = output[:i,:]
                break


    assert output.ndim == 2 and output.shape[1] == 3

    return output # shape: (?,3), 3:XY,Z,t, include N points

def predict2d_TF(curve :np.array,
                 model,
                 mean: torch.Tensor,
                 std: torch.Tensor,
                 out_time: float,
                 fps: float,
                 touch_ground_stop=True,
                 device=torch.device('cpu')):
    
    # curve shape: (N,3), 3:XY,Z,t, pad: 0
    assert curve.ndim == 2 and curve.shape[1] == 3

    model.eval()

    # Remove x,y where is 0 TODO
    # curve = curve[curve[:,[0,1]].any(axis=1)]

    curve_tensor = torch.tensor(curve,dtype=torch.float).unsqueeze(0).to(device)

    pr = []

    inp = (torch.diff(curve_tensor[:,:,[0,1]],dim=1) - mean.to(device)) / std.to(device)
    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
        device)
    dec_inp=start_of_seq

    for i in range(int(fps*out_time)):
        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
        out = model(inp, dec_inp, src_att, trg_att)
        dec_inp=torch.cat((dec_inp,out[:,-1:,:]),1)

    preds_tr_b=((dec_inp[:,1:,[0,1]]*std.to(device)+mean.to(device)).cumsum(1) + curve_tensor[:,-1:,[0,1]]).cpu().numpy()
    pr.append(preds_tr_b)

    pr = np.concatenate(pr, 0).squeeze(0) # batch size = 1

    pad_t = np.expand_dims(np.arange(0,pr.shape[0]) * (1/fps) + curve[-1,2] + 1/fps, axis=1)

    pr = np.concatenate((pr,pad_t),axis=1)

    output = np.concatenate((curve, pr), axis=0)

    if touch_ground_stop:
        for i in range(output.shape[0]):
            if output[i][1] <= 0:
                output = output[:i,:]
                break

    
    assert output.ndim == 2 and output.shape[1] == 3

    return output # shape: (?,3), 3:XY,Z,t, include N points



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
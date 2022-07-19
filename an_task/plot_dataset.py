import argparse
from matplotlib import colors
import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

from pandas import Timestamp
sys.path.append(f"../lib")
sys.path.append(f"../Model3D")
sys.path.append(f"../RNN")
from point import Point, load_points_from_csv, save_points_to_csv
from utils.error_function import space_err, time_err, space_time_err
from physic_model import physics_predict3d
from dataloader import RNNDataSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Draw Dataset')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    # parser.add_argument('--fps', type=str, required=True, help = 'FPS')
    parser.add_argument('--no_smooth', action="store_true", help = 'Show each trajectory')
    parser.add_argument('--each', action="store_true", help = 'Show each trajectory')
    parser.add_argument('--scatter', action="store_true", help = 'N-split scatter')
    parser.add_argument('--heatmap', action="store_true", help = 'N-split heatmap')
    parser.add_argument('--no_show', action="store_true", help = 'No plt.show()')
    args = parser.parse_args()
    # Dataset
    DATASET = args.folder # '../trajectories_dataset/split'
    # FPS = float(args.fps)
    N = 5
    dataset = RNNDataSet(dataset_path=DATASET, N=N, smooth_2d=not args.no_smooth)

    if args.each:
        dataset.show_each()
        sys.exit(1)

    if args.heatmap:
        # heatmap hyperparameters
        x_min = -90
        x_max = 90
        y_min = 0
        y_max = 250
        x_bins = int((x_max-x_min) / 5)
        y_bins = int((y_max-y_min) / 10)

    angle = [(-180,180),(-90,-80),(-80,-60),(-60,-30),(-30,-15),(-15,0),(0,15),(15,30),(30,45),(45,60),(60,80),(80,90)]

    for start_a, end_a in angle:
        cnt = 0
        fig = plt.figure()
        ax_whole = fig.add_subplot(2,2,1)
        ax_dis = fig.add_subplot(2,2,2)
        ax_N = fig.add_subplot(2,2,3)
        ax_N_dis = fig.add_subplot(2,2,4)

        if args.heatmap:
            fig2, ax2 = plt.subplots()
            ax2.set_xlabel("Elevation(degree)")
            ax2.set_ylabel("Speed(km/hr)")
            ax2.set_title(f"{start_a}° ~ {end_a}° Distribution Split by each {N}")
            elevation_lst = []
            speed_lst = []
            ax2_cbar = fig2.add_axes([0.92, 0.15, 0.02, 0.5])

        for idx, data in dataset.whole_2d().items():
            elevation = math.degrees(math.atan2(data[1,1]-data[0,1], data[1,0]-data[0,0]))
            speed = np.linalg.norm(data[1,:-1]-data[0,:-1]) / (data[1,-1]-data[0,-1])

            if elevation >= start_a and elevation < end_a:
                tmp = data.copy()
                tmp -= tmp[0] # move to origin, reset time to zero
                ax_whole.plot(tmp[:,0],tmp[:,1],marker='o',markersize=2)
                ax_dis.scatter(elevation,speed*3600/1000,marker='o',s=12)
                cnt += 1
            # For each N as a slice
            for i in range(data.shape[0]-N+1):
                tmp = data[i:i+N].copy()
                tmp -= tmp[0] # move to origin, reset time to zero
                elevation = math.degrees(math.atan2(tmp[1,1]-tmp[0,1], tmp[1,0]-tmp[0,0]))
                if((tmp[1,-1]-tmp[0,-1]) == 0.0):
                    # timestamp same!
                    continue
                speed = np.linalg.norm(tmp[1,:-1]-tmp[0,:-1]) / (tmp[1,-1]-tmp[0,-1])

                if elevation >= start_a and elevation < end_a:
                    ax_N.plot(tmp[:,0],tmp[:,1],marker='o',markersize=2)
                    if args.heatmap and elevation < x_max and elevation > x_min and speed < y_max and speed > y_min:
                        elevation_lst.append(elevation)
                        speed_lst.append(speed)
                    if args.scatter:
                        ax_N_dis.scatter(elevation,speed*3600/1000,marker='o',s=6)


                    # DEBUG strange datas
                    if elevation > 90:
                        print(f"e:{elevation}, #{idx}:{i}-th")
                    elif elevation < 0 and speed*3600/1000 > 300:
                        print(f"e:{elevation},s:{speed*3600/1000}, #{idx}:{i}-th")


        ax_whole.set_xlabel("Distance")
        ax_whole.set_ylabel("Height")
        ax_whole.set_title(f"{start_a}° ~ {end_a}°")
        ax_dis.set_xlabel("Elevation(degree)")
        ax_dis.set_ylabel("Speed(km/hr)")
        ax_dis.set_title(f"{start_a}° ~ {end_a}° Distribution")

        ax_N.set_xlabel("Distance")
        ax_N.set_ylabel("Height")
        ax_N.set_title(f"{start_a}° ~ {end_a}° Split by each {N}")
        ax_N_dis.set_xlabel("Elevation(degree)")
        ax_N_dis.set_ylabel("Speed(km/hr)")
        ax_N_dis.set_title(f"{start_a}° ~ {end_a}° Distribution Split by each {N}")

        if args.heatmap:
            h = ax2.hist2d(elevation_lst,[s*3600/1000 for s in speed_lst],bins=[x_bins,y_bins], norm=colors.LogNorm(), range=[[x_min,x_max],[y_min,y_max]], cmap='YlGn')
            cbar = fig2.colorbar(h[3],cax=ax2_cbar, format="%0.1f")

        if not args.no_show:
            plt.show()
        print(f"{start_a}° ~ {end_a}° Count: {cnt}")
        # output_folder = "./dataset_figure"
        # os.makedirs(output_folder,exist_ok=True)
        # plt.savefig(os.path.join(output_folder, f"{start_a}_{end_a}.png"))

import argparse
import json
import logging
import os
import sys
import threading
import time
import csv
from unicodedata import bidirectional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from datetime import datetime

sys.path.append(f"../lib")
sys.path.append(f"../Model3D")
sys.path.append(f"../RNN")
from point import Point, load_points_from_csv, save_points_to_csv
from error_function import space_err, time_err, space_time_err
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from physic_model import physics_predict3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Validate Physics Model and Find Good coeffs')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument('--smooth', action="store_true", help = 'Smooth when projecting to 2d')
    args = parser.parse_args()
    # Dataset
    DATASET = args.folder
    # FPS
    fps = args.fps

    # Use which points to physically predict
    if "vicon" in args.folder:
        FIRST_POINT_IDX = 0
        SECOND_POINT_IDX = 1
    else:
        FIRST_POINT_IDX = 1
        SECOND_POINT_IDX = 2
        # SECOND_POINT_IDX = int(0.05*fps) # 50ms

    print(f"First: {FIRST_POINT_IDX}, SECOND POINT IDX: {SECOND_POINT_IDX}")

    # Load Dataset
    trajectories_3d2d = []
    trajectories_3d = []
    for folder in sorted(os.listdir(DATASET)):
        if os.path.exists(os.path.join(DATASET, folder, 'vicon.csv')):
            csv_3d = os.path.join(DATASET, folder, 'vicon.csv')
        elif os.path.exists(os.path.join(DATASET, folder, 'Model3D.csv')):
            csv_3d = os.path.join(DATASET, folder, 'Model3D.csv')
        else:
            sys.exit(1)
        points_3d = load_points_from_csv(csv_3d)
        points_3d = [p for p in points_3d if p.visibility == 1]
        tra_3d = np.stack([p.toXYZT() for p in points_3d], axis=0)
        trajectories_3d.append(tra_3d)
        tra_2d, tra_3d2d, slope, intercept = FitVerticalPlaneTo2D(tra_3d, smooth_2d=args.smooth)
        trajectories_3d2d.append(tra_3d2d)
        # points_3d2d = [Point(visibility=1,x=i[0],y=i[1],z=i[2],timestamp=i[3]) for i in tra_3d2d]
        # save_points_to_csv(points_3d2d, os.path.join(DATASET, folder, '3d2d.csv'))

        # points_physics_3d2d = [Point(visibility=1,x=i[0],y=i[1],z=i[2],timestamp=i[3]) for i in tra_physics_3d2d]
        # save_points_to_csv(points_physics_3d2d, os.path.join(DATASET, folder, 'physic_3d2d.csv'))


    alpha_default = 0.2152
    g_default = 9.81



    ##### Find the best coeff for the dataset #####

    dx, dy = 0.002, 0.05
    y, x = np.mgrid[slice(g_default-8.0, g_default+8.0 + dy, dy),
                    slice(alpha_default-0.07, alpha_default+0.07 + dx, dx)]
    z_space_2d = np.zeros(shape=x.shape,dtype=float)
    z_space_time_2d = np.zeros(shape=x.shape,dtype=float)
    z_space_time_2d_each = np.zeros(shape=(x.shape[0],x.shape[1],len(trajectories_3d2d)),dtype=float)
    z_space_3d = np.zeros(shape=x.shape,dtype=float)
    z_space_time_3d = np.zeros(shape=x.shape,dtype=float)

    print(f"({x.shape[0],x.shape[1]})")
    # print("RMSE")

    for i in range(x.shape[0]):
        print(i)
        for j in range(x.shape[1]):
            alpha = x[i][j]
            g = y[i][j]
            space_2d = []
            space_time_2d = []
            space_3d = []
            space_time_3d = []
            for k, tra_3d2d in enumerate(trajectories_3d2d):

                tra_physics_3d2d = physics_predict3d(tra_3d2d[FIRST_POINT_IDX],tra_3d2d[SECOND_POINT_IDX],alpha=alpha,g=g)

                # space_2d.append(space_err(tra_3d2d[SECOND_POINT_IDX+1:], tra_physics_3d2d[2:]))
                space_time_2d.append(space_time_err(tra_3d2d[SECOND_POINT_IDX+1:], tra_physics_3d2d[2:]))

                # space_3d.append(space_err(trajectories_3d[k][SECOND_POINT_IDX+1:], tra_physics_3d2d[2:]))
                # space_time_3d.append(space_time_err(trajectories_3d[k][SECOND_POINT_IDX+1:], tra_physics_3d2d[2:]))

                z_space_time_2d_each[i][j][k] = space_time_err(tra_3d2d[SECOND_POINT_IDX+1:], tra_physics_3d2d[2:])

            space_time_2d = np.stack(space_time_2d)
            #space_time_3d = np.stack(space_time_3d)

            # z_space_2d[i][j] = np.nanmean(space_2d)

            
            z_space_time_2d[i][j] = np.nanmean(space_time_2d)

            # z_space_3d[i][j] = np.nanmean(space_3d)
            # z_space_time_3d[i][j] = np.sqrt(np.nanmean((space_time_3d)**2))



    """
    ##### Space 2D
    plt.axvline(x=alpha_default, color='k', linestyle='--')
    plt.axhline(y=g_default, color='k', linestyle='--')
    plt.xlabel("Air drag acceleration")
    plt.ylabel("Gravity")
    z_space_2d = z_space_2d[:-1,:-1] 
    z_min, z_max = z_space_2d.min(), z_space_2d.max() 
    c = plt.pcolormesh(x, y, z_space_2d, cmap ='hot', vmin = z_min, vmax = z_max)
    plt.title(f"Space 2D Error(m). Best(a,g) at ({x[np.unravel_index(z_space_2d.argmin(), z_space_2d.shape)]:.3f},{y[np.unravel_index(z_space_2d.argmin(), z_space_2d.shape)]:.3f})")
    plt.colorbar(c)
    plt.scatter(x[np.unravel_index(z_space_2d.argmin(), z_space_2d.shape)], y[np.unravel_index(z_space_2d.argmin(), z_space_2d.shape)],c='red')
    plt.show()
    """
    ##### Space Time 2D
    plt.axvline(x=alpha_default, color='k', linestyle='--')
    plt.axhline(y=g_default, color='k', linestyle='--')
    plt.xlabel("Air drag acceleration")
    plt.ylabel("Gravity")
    z_space_time_2d = z_space_time_2d[:-1,:-1]
    z_space_time_2d_each = z_space_time_2d_each[:-1,:-1,:]
    z_min, z_max = z_space_time_2d.min(), z_space_time_2d.max()
    print(f"{z_min} {z_max}")
    c = plt.pcolormesh(x, y, z_space_time_2d, cmap ='hot', vmin = z_min, vmax = z_max)
    plt.title(f"Space Time 2D Error(m). Best(a,g) at ({x[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)]:.3f},{y[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)]:.3f})")
    print(f"Best(a,g) at ({x[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)]:.3f},{y[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)]:.3f})")
    plt.colorbar(c)
    plt.scatter(x[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)], y[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)],c='red')

    for k in range(z_space_time_2d_each.shape[2]):
        plt.scatter(x[np.unravel_index(z_space_time_2d_each[:,:,k].argmin(), z_space_time_2d.shape)], y[np.unravel_index(z_space_time_2d_each[:,:,k].argmin(), z_space_time_2d.shape)],c='blue',s=2)

    #for i in range(z_space_time_2d_each.shape[0]):
    #    plt.scatter(x[])
    
    plt.show()
    """
    ##### Space 3D
    plt.axvline(x=alpha_default, color='k', linestyle='--')
    plt.axhline(y=g_default, color='k', linestyle='--')
    plt.xlabel("Air drag acceleration")
    plt.ylabel("Gravity")
    z_space_3d = z_space_3d[:-1,:-1] 
    z_min, z_max = z_space_3d.min(), z_space_3d.max()
    c = plt.pcolormesh(x, y, z_space_3d, cmap ='hot', vmin = z_min, vmax = z_max)
    plt.title(f"Space 3D Error(m). Best(a,g) at ({x[np.unravel_index(z_space_3d.argmin(), z_space_3d.shape)]:.3f},{y[np.unravel_index(z_space_3d.argmin(), z_space_3d.shape)]:.3f})")
    plt.colorbar(c)
    plt.scatter(x[np.unravel_index(z_space_3d.argmin(), z_space_3d.shape)], y[np.unravel_index(z_space_3d.argmin(), z_space_3d.shape)],c='red')
    plt.show()
    
    ##### Space Time 3D
    plt.axvline(x=alpha_default, color='k', linestyle='--')
    plt.axhline(y=g_default, color='k', linestyle='--')
    plt.xlabel("Air drag acceleration")
    plt.ylabel("Gravity")
    z_space_time_3d = z_space_time_3d[:-1,:-1] 
    z_min, z_max = z_space_time_3d.min(), z_space_time_3d.max()
    c = plt.pcolormesh(x, y, z_space_time_3d, cmap ='hot', vmin = z_min, vmax = z_max)
    plt.title(f"Space Time 3D Error(m). Best(a,g) at ({x[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)]:.3f},{y[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)]:.3f})")
    print(f"Best(a,g) at ({x[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)]:.3f},{y[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)]:.3f})")
    plt.colorbar(c)
    plt.scatter(x[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)], y[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)],c='red')
    plt.show()
    """

    # width = 0.15

    # x = np.arange(len(folder_name))

    # plt.bar(x, z_space_2d, width, color='orange', label='Space 2D')
    # plt.bar(x + width, z_space_time_2d, width, color='red', label='Space & Time 2D')
    # plt.bar(x + width*2, z_space_3d, width, color='green', label='Space 3D')
    # plt.bar(x + width*3, z_time_3d, width, color='blue', label='Time 3D')
    # plt.bar(x + width*4, z_space_time_3d, width, color='pink', label='Space & Time 3D')
    # plt.xticks(x + width*2, folder_name)
    # plt.ylabel('error(meter)')
    # plt.title('Vicon & Physic Model')
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')

    # plt.show()
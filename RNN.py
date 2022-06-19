"""
RNN : To receive trajectory of EventDetector and publisher predicted trajectory
"""
import threading
import sys
import cv2
import logging
import configparser
import queue
import base64
import paho.mqtt.client as mqtt
import numpy as np
import os
import json
import torch
import math
import argparse
import pickle
import time

from datetime import datetime
from typing import Optional
from sklearn.metrics import confusion_matrix

# Our System's library
from threeDprojectTo2D import FitVerticalPlaneTo2D
from blstm import Blstm
from predict import predict3d, predict2d
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append("../lib")
from common import load_config
from inspector import sendPerformance, sendNodeStateMsg
from point import Point, sendPoints
from receiver import PointReceiver

class Predictor(threading.Thread):
    def __init__(self, name, broker, topic, settings):
        threading.Thread.__init__(self)

        self.name = name
        IN_SIZE = int(settings['IN_SIZE'])
        OUT_SIZE = int(settings['OUT_SIZE'])
        self.HIDDEN_SIZE = int(settings['HIDDEN_SIZE'])
        self.HIDDEN_LAYER = int(settings['HIDDEN_LAYER'])
        self.BATCH_SIZE = int(settings['BATCH_SIZE'])
        self.nNumber = int(settings['nNumber'])
        self.pNumber = int(settings['pNumber'])
        weight = settings['weight']
        Learning_RATE = 0.01
        self.inputs = []
        self.lock = threading.Lock()
        self.rnn = MyRnn(in_size=IN_SIZE, out_size=OUT_SIZE, hidden_size=self.HIDDEN_SIZE, hidden_layer=self.HIDDEN_LAYER)
        self.rnn.load_state_dict(torch.load(weight))

        self.topic = topic
        # setup MQTT client
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.connect(broker)
        self.client = client

        self.alive = False

    def on_connect(self, client, userdata, flag, rc):
        logging.info("Connected with result code: {}".format(rc))

    def stop(self):
        self.alive = False

    def publishTrajectory(self, output_trajectory):
        logging.info("Predict {} Points:".format(len(output_trajectory)))
        for p in output_trajectory:
            logging.debug("Predict id:{:>4},x={:>6.3f},y={:>6.3f},z={:>6.3f}".format(p.fid,p.x,p.y,p.z))

        timestamp = datetime.now().timestamp()
        sendPoints(self.client, self.topic, output_trajectory)

    def insertList(self, list):
        with self.lock:
            self.inputs.extend(list)
        logging.info("Receive {} Points:".format(len(self.inputs)))
        for p in self.inputs:
            logging.debug("insert id:{:>4},x={:>6.3f},y={:>6.3f},z={:>6.3f}".format(p.fid,p.x,p.y,p.z))

    def run(self):
        logging.info("Predictor is started.")
        self.alive = True
        N = self.nNumber
        PREDICT_NUMBER = self.pNumber
        while self.alive:
            if len(self.inputs) >= N:
                npoints = []
                for i in range(N):
                    npoints.append(np.array([self.inputs[i].x,self.inputs[i].y,self.inputs[i].z]))
                npoints = np.stack(npoints, axis=0)
                #######################################################################################################
                # Project N points on LST Plane and to XZ Plane


                output = predict3d(npoints, self.rnn, self.HIDDEN_LAYER, self.HIDDEN_SIZE, max_predict_number=PREDICT_NUMBER)

                output_trajectory = []
                for idx in range(output.shape[0]):
                    output_trajectory.append(Point(fid=self.inputs[0].fid+N+idx,
                                                   timestamp=0.0,
                                                   x=output[idx][0],
                                                   y=output[idx][1],
                                                   z=output[idx][2],
                                                   color='yellow')) # TODO timestamp


                # Put Points to Queue and it will Publish
                self.publishTrajectory(output_trajectory)
                fids = []
                for i in range(N):
                    fids.append(self.inputs[i].fid)

                with self.lock:
                    del self.inputs[0:N]
            else:
                time.sleep(0.016)
        logging.info("Predictor is terminated.")

class MainThread(threading.Thread):
    def __init__(self, args, settings):
        threading.Thread.__init__(self)

        self.nodename = args.nodename
        queue_size = int(settings['queue_size'])

        broker = settings['mqtt_broker']
        input_topic = settings['input_topic']
        output_topic = settings['output_topic']

        # Setup N Points Receiver
        self.receiver = PointReceiver(self.nodename, broker, input_topic, queue_size)

        # Setup Predictor
        self.predictor = Predictor(self.nodename, broker, output_topic, settings)

        # setup MQTT client
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(broker)
        self.client = client

        self.killswitch = threading.Event()

        self.fps = 30 # TODO

    def on_connect(self, client, userdata, flag, rc):
        logging.info("Connected with result code: {}".format(rc))
        self.client.subscribe(self.nodename) # TODO
        self.client.subscribe('system_control')

    def on_message(self, client, userdata, msg):
        cmds = json.loads(msg.payload)
        if msg.topic == 'system_control':
            if 'stop' in cmds:
                if cmds['stop'] == True:
                    self.stop()
        else:
            if 'stop' in cmds:
                if cmds['stop'] == True:
                    self.stop()

    def stop(self):
        self.alive = False

    def run(self):
        logging.debug("started.")

        self.alive = True
        self.client.loop_start()
        self.receiver.start()
        self.predictor.start()

        logging.info("{} is ready.".format(self.nodename))
        sendNodeStateMsg(self.client, self.nodename, "ready")

        while self.alive:
            if self.receiver.queue:
                self.predictor.insertList(self.receiver.queue)
                self.receiver.queue.clear()
            else:
                time.sleep(1/self.fps)

        self.predictor.stop()
        self.predictor.join()
        self.receiver.stop()
        self.receiver.join()
        sendNodeStateMsg(self.client, self.nodename, "terminated")
        self.client.loop_stop()

        logging.info("Main Thread terminated.")

def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = 'RNN')
    parser.add_argument('--project', type=str, required=True, help = 'project name')
    parser.add_argument('--nodename', type=str, default='RNN', help = 'mqtt node name (default: RNN)')
    args = parser.parse_args()

    return args

def main():
    # Parse arguments
    args = parse_args()
    # Load configs+
    projectCfg = f"{ROOTDIR}/projects/{args.project}.cfg"
    settings = load_config(projectCfg, args.nodename)
    # Start MainThread
    mainThread = MainThread(args, settings)
    mainThread.start()
    mainThread.join()

if __name__ == '__main__':
    main()

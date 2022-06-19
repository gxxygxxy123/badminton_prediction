import os
import sys
import csv
import json
import threading
import argparse
import logging
import paho.mqtt.client as mqtt
import numpy as np

from typing import Optional

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)

sys.path.append(f"{ROOTDIR}/RNN")
from predict import predict3d
from physic_model import physics_predict3d
from blstm import Blstm

sys.path.append(f"{ROOTDIR}/lib")
from common import loadNodeConfig
from point import Point
from writer import CSVWriter
from inspector import sendNodeStateMsg

class RnnPredictorThread(threading.Thread):
    def __init__(self, file_path):
        threading.Thread.__init__(self)
        self.killswitch = threading.Event()

        self.file_path = file_path
        self.dir = os.path.dirname(self.file_path)
        self.rnn = Blstm()
        self.track = self.getFivePoints()
        self.csvWriter = CSVWriter(name='RnnPredictor', filename=f"{self.dir}/Predict3d.csv")

    def stop(self):
        self.killswitch.set()

    def run(self):
        # get csv file path and output 3D predict trajectory
        # output = predict3d(self.track, self.rnn)
        output = physics_predict3d(self.track[0], self.track[1])
        frame = 0
        final = output.shape[0]
        for point in output:
            if frame == 0:
                self.csvWriter.writePoints(Point(fid=frame, x=point[0], y=point[1], z=point[2], event=1))
            elif frame == final - 1:
                self.csvWriter.writePoints(Point(fid=frame, x=point[0], y=point[1], z=point[2], event=3))
            else:
                self.csvWriter.writePoints(Point(fid=frame, x=point[0], y=point[1], z=point[2]))
            frame += 1

    def getFivePoints(self):
        track = []
        c = 0
        start = False
        with open(self.file_path, 'r', newline='') as csvFile:
            rows = csv.DictReader(csvFile)
            for row in rows:
                if int(row['Event']) == 1:
                    start = True
                    continue
                if start:
                    if c > 5:
                        break
                    else:
                        c += 1
                        point = Point(fid=row['Frame'], timestamp=row['Timestamp'], visibility=row['Visibility'], x=row['X'], y=row['Y'], z=row['Z'],event=row['Event']).toXYZT()
                        track.append(point)
                else:
                    continue
        return np.array(track)

class MainThread(threading.Thread):
    def __init__(self, args, info):
        threading.Thread.__init__(self)
        self.killswitch = threading.Event()

        # setup configuration
        self.info = info
        self.project_name = args.project
        self.nodename = args.nodename

        # setup MQTT client
        self.input_topic = self.info['input_topic']
        self.broker = self.info['mqtt_broker']
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(self.broker)
        self.client = client

        # setup data publisher
        self.output_topic = self.info['output_topic']

        # initial
        self.input_files = []
        self.rnnThreads = []
        self.threads_size = int(self.info['threads_size'])

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"{self.nodename} Connected with result code: {str(rc)}")
        self.client.subscribe(self.nodename)
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
            elif 'file_path' in cmds:
                file_path = cmds['file_path']
                print(file_path)
                self.input_files.append(file_path)

    def stop(self):
        for i in range(len(self.rnnThreads)):
            self.rnnThreads[i].join()
        self.alive = False
        print('RnnPredictor Finished')

    def run(self):
        logging.debug("{} started.".format(self.nodename))

        self.client.loop_start()

        logging.info(f"{self.nodename} is ready.")
        sendNodeStateMsg(self.client, self.nodename, "ready")
        self.alive = True

        while self.alive:
            if len(self.rnnThreads) < self.threads_size and len(self.input_files) > 0:
                print('start thread')
                file_path = self.input_files[0]
                del self.input_files[0]
                rnnThread = RnnPredictorThread(file_path=file_path)
                rnnThread.start()
                self.rnnThreads.append(rnnThread)
            else:
                for i in range(len(self.rnnThreads)):
                    if self.rnnThreads[i].is_alive() == False:
                        del self.rnnThreads[i]
                        break

        for i in range(len(self.rnnThreads)):
            self.rnnThreads[i].join()
        sendNodeStateMsg(self.client, self.nodename, "terminated")


def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = 'RnnPredictor')
    parser.add_argument('--project', type=str, default = 'coachbox', help = 'project name (default: coachbox)')
    parser.add_argument('--nodename', type=str, default = 'RnnPredictor', help = 'mqtt node name (default: RnnPredictor)')
    args = parser.parse_args()

    return args

def main():
    # Parse arguments
    args = parse_args()
    # Load configs
    projectCfg = f"{ROOTDIR}/config"
    settings = loadNodeConfig(projectCfg, args.nodename)

    # Start MainThread
    mainThread = MainThread(args, settings)
    mainThread.start()
    mainThread.join()

if __name__ == '__main__':
    main()
    # mosquitto_pub -h localhost -t RnnPredictor -m "{\"file_path\": \"../replay/20210321/1/Model3D.csv\"}"

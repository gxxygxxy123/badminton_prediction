import sys
from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QLabel, QSpinBox, \
    QCheckBox, QGridLayout, QPlainTextEdit, QSizePolicy, QRadioButton, QButtonGroup
import math
import numpy as np
import copy
from dataloader import RNNDataSet



# Subclass QMainWindow to customize your application's main window
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RNN Dataset Inspector")
        self.setFixedSize(QSize(1800, 900))

        hlayout = QHBoxLayout()
        self.setLayout(hlayout)

        self.fig = Figure(figsize=(15,6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedSize(QSize(1400, 800))
        self.insert_ax()

        TRAIN_DATASET = '../trajectories_dataset/train'
        TEST_DATASET = '../trajectories_dataset/test'
        self.N = 5
        self.fps = 60
        self.train_smth_dataset = RNNDataSet(dataset_path=TRAIN_DATASET, N=self.N, smooth_2d=True, comment="Train")
        self.train_nosmth_dataset = RNNDataSet(dataset_path=TRAIN_DATASET, N=self.N, smooth_2d=False, comment="Train")

        self.test_smth_dataset = RNNDataSet(dataset_path=TEST_DATASET, N=self.N, smooth_2d=True, comment="Test")
        self.test_nosmth_dataset = RNNDataSet(dataset_path=TEST_DATASET, N=self.N, smooth_2d=False, comment="Test")

        vlayout = QVBoxLayout()
        speed_layout = QHBoxLayout()
        elevation_layout = QHBoxLayout()
        slice_layout = QHBoxLayout()
        dis_layout = QHBoxLayout()
        dataset_layout = QHBoxLayout()
        data_layout = QHBoxLayout()
        vlayout.addLayout(speed_layout)
        vlayout.addLayout(elevation_layout)
        vlayout.addLayout(slice_layout)
        vlayout.addLayout(dis_layout)
        vlayout.addLayout(dataset_layout)
        vlayout.addLayout(data_layout)
        # Speed
        self.speed_text1 = QLabel("初速度：")
        self.start_speed = QSpinBox()
        self.start_speed.setRange(0,10000)
        self.start_speed.setValue(0)
        self.speed_text2 = QLabel("到")
        self.end_speed = QSpinBox()
        self.end_speed.setRange(0,10000)
        self.end_speed.setValue(500)
        self.speed_text3 = QLabel("km/hr")
        speed_layout.addWidget(self.speed_text1)
        speed_layout.addWidget(self.start_speed)
        speed_layout.addWidget(self.speed_text2)
        speed_layout.addWidget(self.end_speed)
        speed_layout.addWidget(self.speed_text3)
        self.start_speed.textChanged.connect(self.update_chart)
        self.end_speed.textChanged.connect(self.update_chart)
        # Elevation
        self.elevation_text1 = QLabel("仰角(-90~90)：")
        self.start_elevation = QSpinBox()
        self.start_elevation.setRange(-90,90)
        self.start_elevation.setValue(-90)
        self.elevation_text2 = QLabel("到")
        self.end_elevation = QSpinBox()
        self.end_elevation.setRange(-90,90)
        self.end_elevation.setValue(90)
        self.elevation_text3 = QLabel("度")
        elevation_layout.addWidget(self.elevation_text1)
        elevation_layout.addWidget(self.start_elevation)
        elevation_layout.addWidget(self.elevation_text2)
        elevation_layout.addWidget(self.end_elevation)
        elevation_layout.addWidget(self.elevation_text3)
        self.start_elevation.textChanged.connect(self.update_chart)
        self.end_elevation.textChanged.connect(self.update_chart)
        # Slice & Move to origin
        self.chk_slice = QCheckBox("每5點分段")
        self.chk_slice.setChecked(False)
        self.chk_origin = QCheckBox("軌跡移至原點")
        self.chk_origin.setChecked(False)
        self.chk_nosmth = QCheckBox("無Smooth原軌跡")
        self.chk_nosmth.setChecked(False)
        slice_layout.addWidget(self.chk_slice)
        slice_layout.addWidget(self.chk_origin)
        slice_layout.addWidget(self.chk_nosmth)
        self.chk_slice.toggled.connect(self.update_chart)
        self.chk_origin.toggled.connect(self.update_chart)
        self.chk_nosmth.toggled.connect(self.update_chart)
        # Trajectories/Distribution
        self.show_btngroup = QButtonGroup(self)
        self.btn_show_trajectory = QRadioButton("軌跡距離/高度圖", self)
        self.btn_show_diversity_scatterplot = QRadioButton("初速仰角散佈圖", self)
        self.btn_show_diversity_heatmap = QRadioButton("初速仰角熱度圖", self)

        self.show_btngroup.addButton(self.btn_show_trajectory)
        self.show_btngroup.addButton(self.btn_show_diversity_scatterplot)
        self.show_btngroup.addButton(self.btn_show_diversity_heatmap)

        dis_layout.addWidget(self.btn_show_trajectory)
        dis_layout.addWidget(self.btn_show_diversity_scatterplot)
        dis_layout.addWidget(self.btn_show_diversity_heatmap)

        self.btn_show_trajectory.toggled.connect(self.update_chart)
        self.btn_show_diversity_scatterplot.toggled.connect(self.update_chart)
        self.btn_show_diversity_heatmap.toggled.connect(self.update_chart)

        self.btn_show_trajectory.toggled.connect(self.toggle_dis)
        self.btn_show_diversity_scatterplot.toggled.connect(self.toggle_dis)
        self.btn_show_diversity_heatmap.toggled.connect(self.toggle_dis)

        
        # Dataset choose
        self.chk_dataset_train = QCheckBox("Train")
        self.chk_dataset_train.setChecked(False)
        self.chk_dataset_test = QCheckBox("Test")
        self.chk_dataset_test.setChecked(False)
        dataset_layout.addWidget(self.chk_dataset_train)
        dataset_layout.addWidget(self.chk_dataset_test)
        self.chk_dataset_train.toggled.connect(self.update_chart)
        self.chk_dataset_test.toggled.connect(self.update_chart)

        # Data Index List
        self.data_idx = QLabel()
        self.data_idx.setFixedSize(QSize(300,500))
        self.data_idx.setWordWrap(True)
        data_layout.addWidget(self.data_idx)


        hlayout.addWidget(self.canvas)
        hlayout.addLayout(vlayout)
        
        # Default Options
        self.btn_show_trajectory.setChecked(True)

        # Update chart
        self.update_chart()

    def insert_ax(self):
        font = {
            'weight': 'normal',
            'size': 16
        }
        matplotlib.rc('font', **font)

        self.ax = self.canvas.figure.subplots()

        self.ax_cbar = self.fig.add_axes([0.92, 0.15, 0.02, 0.5])
        self.ax_cbar.set_visible(False)
        


    def train_test_bbox(self):
        self.ax.text(0.05,0.95, 'Train', ha='center', color='black', weight='bold', transform = self.ax.transAxes)
        self.ax.text(0.05,0.9, 'Test', ha='center', color='red', weight='bold', transform = self.ax.transAxes)

    def update_chart(self):
        self.ax.clear()

        self.data_idx.clear()
        # for i in reversed(range(self.data_layout.count())): 
        #     self.data_layout.itemAt(i).widget().deleteLater()
        # self.ax.set_ylim([-1, 4])
        # self.ax.set_xlim([-1, 14])

        # heatmap hyperparameters
        x_min = -90
        x_max = 90
        y_min = 0
        y_max = 200
        x_bins = int((x_max-x_min) / 5)
        y_bins = int((y_max-y_min) / 10)
        if self.btn_show_diversity_scatterplot.isChecked() or self.btn_show_diversity_heatmap.isChecked():
            self.ax.set_xlabel("Elevation(degree)")
            self.ax.set_ylabel("Initial Velocity(km/hr)")
            self.ax.set_xlim([x_min, x_max])
            self.ax.set_ylim([y_min, y_max])
        elif self.btn_show_trajectory.isChecked():
            self.ax.set_xlabel("Distance(m)")
            self.ax.set_ylabel("Height(m)")
            if not self.chk_slice.isChecked() and not self.chk_origin.isChecked():
                self.ax.set_xlim([-1, 14])
                self.ax.set_ylim([0, 4])

        # heatmap
        list_elevation = []
        list_speed = []

        ### Train
        train_data_cnt = 0

        train_data_smth = {}
        train_data_nosmth = {}
        if self.chk_dataset_train.isChecked():
            if self.chk_slice.isChecked():
                train_data_smth = copy.deepcopy(self.train_smth_dataset.slice())
                if self.chk_nosmth.isChecked():
                    train_data_nosmth = copy.deepcopy(self.train_nosmth_dataset.slice())
            else:
                train_data_smth = copy.deepcopy(self.train_smth_dataset.whole_2d())
                if self.chk_nosmth.isChecked():
                    train_data_nosmth = copy.deepcopy(self.train_nosmth_dataset.whole_2d())
            if self.chk_origin.isChecked():
                for idx, data in train_data_smth.items():
                    data -= data[0]
                if self.chk_nosmth.isChecked():
                    for idx, data in train_data_nosmth.items():
                        data -= data[0]

            for idx, data in train_data_smth.items():
                elevation = math.degrees(math.atan2(data[1,1]-data[0,1], data[1,0]-data[0,0]))
                speed = np.linalg.norm(data[1,:]-data[0,:])*self.fps*3600/1000
                if not (elevation >= self.start_elevation.value() and elevation <= self.end_elevation.value()):
                    continue
                if not (speed >= self.start_speed.value() and speed <= self.end_speed.value()):
                    continue

                if self.btn_show_diversity_scatterplot.isChecked():
                    if self.chk_dataset_test.isChecked():

                        self.ax.scatter(elevation,speed,marker='o',s=8, color='black')
                        self.train_test_bbox()
                    else:
                        self.ax.scatter(elevation,speed,marker='o',s=8)

                elif self.btn_show_trajectory.isChecked():
                    if self.chk_dataset_test.isChecked():

                        p = self.ax.plot(data[:,0],data[:,1],marker='o',markersize=4, color='black')
                        self.train_test_bbox()
                    else:
                        p = self.ax.plot(data[:,0],data[:,1],marker='o',markersize=4)
                    if self.chk_nosmth.isChecked():
                        self.ax.plot(train_data_nosmth[idx][:,0],train_data_nosmth[idx][:,1],marker='o',markersize=4,alpha=0.3,color = p[0].get_color())

                train_data_cnt += 1
                list_elevation.append(elevation)
                list_speed.append(speed)
                self.data_idx.setText(self.data_idx.text()+f"{idx}, ")
        
        ### Test
        test_data_cnt = 0

        test_data_smth = {}
        test_data_nosmth = {}
        if self.chk_dataset_test.isChecked():
            if self.chk_slice.isChecked():
                test_data_smth = copy.deepcopy(self.test_smth_dataset.slice())
                if self.chk_nosmth.isChecked():
                    test_data_nosmth = copy.deepcopy(self.test_nosmth_dataset.slice())
            else:
                test_data_smth = copy.deepcopy(self.test_smth_dataset.whole_2d())
                if self.chk_nosmth.isChecked():
                    test_data_nosmth = copy.deepcopy(self.test_nosmth_dataset.whole_2d())
            if self.chk_origin.isChecked():
                for idx, data in test_data_smth.items():
                    data -= data[0]
                if self.chk_nosmth.isChecked():
                    for idx, data in test_data_nosmth.items():
                        data -= data[0]
            for idx, data in test_data_smth.items():
                elevation = math.degrees(math.atan2(data[1,1]-data[0,1], data[1,0]-data[0,0]))
                speed = np.linalg.norm(data[1,:]-data[0,:])*self.fps*3600/1000
                if not (elevation >= self.start_elevation.value() and elevation <= self.end_elevation.value()):
                    continue
                if not (speed >= self.start_speed.value() and speed <= self.end_speed.value()):
                    continue
                if self.btn_show_diversity_scatterplot.isChecked():
                    if self.chk_dataset_train.isChecked():
                        self.ax.scatter(elevation,speed,marker='o',s=8, color='red')
                    else:
                        self.ax.scatter(elevation,speed,marker='o',s=8)

                elif self.btn_show_trajectory.isChecked():
                    if self.chk_dataset_train.isChecked():
                        p = self.ax.plot(data[:,0],data[:,1],marker='o',markersize=4, color='red')
                    else:
                        p = self.ax.plot(data[:,0],data[:,1],marker='o',markersize=4)
                    if self.chk_nosmth.isChecked():
                        self.ax.plot(test_data_nosmth[idx][:,0],test_data_nosmth[idx][:,1],marker='o',markersize=4,alpha=0.3,color = p[0].get_color())

                test_data_cnt += 1
                list_elevation.append(elevation)
                list_speed.append(speed)
                self.data_idx.setText(self.data_idx.text()+f"{idx}, ")

        # heatmap
        if self.btn_show_diversity_heatmap.isChecked() and (self.chk_dataset_train.isChecked() or self.chk_dataset_test.isChecked()):
            h = self.ax.hist2d(list_elevation,list_speed,bins=[x_bins,y_bins], norm=colors.LogNorm(), range=[[x_min,x_max],[y_min,y_max]], cmap='RdYlGn_r')
            cbar = self.fig.colorbar(h[3],cax=self.ax_cbar, format="%0.1f")


        self.ax.set_title(f"Dataset, Train: {train_data_cnt}, Test: {test_data_cnt}")
        self.canvas.draw()

    def toggle_dis(self):
        if self.btn_show_diversity_scatterplot.isChecked() or self.btn_show_diversity_heatmap.isChecked():
            self.chk_origin.setChecked(False)
            self.chk_origin.setEnabled(False)
            self.chk_nosmth.setChecked(False)
            self.chk_nosmth.setEnabled(False)
        else:
            self.chk_origin.setEnabled(True)
            self.chk_nosmth.setEnabled(True)

        if not self.btn_show_diversity_heatmap.isChecked():
            self.ax_cbar.set_visible(False)
        else:
            self.ax_cbar.set_visible(True)
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
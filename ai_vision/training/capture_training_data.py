import argparse
from logging import exception
import random
import sys
import os
import time
import json
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import jetson.utils
import datetime
from training_helpers import *
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstVideo
import threading

# parse the command line
parser = argparse.ArgumentParser(description = "capture images and store them in folders according to the configuration",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--manual_mode", type=bool, default=False, help="choose manual mode if you want to manually choose chambers and colors")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

dirname = os.path.dirname(os.path.realpath(__file__))
dirname = os.path.dirname(dirname)

# load settings from network config file
with open(os.path.join(dirname, "ressource", "neural_network_config.json")) as f:
    neural_network_config = json.load(f)
video_source = neural_network_config["video_source"]
video_sink = neural_network_config["video_sink"]
width = neural_network_config["width"]
height = neural_network_config["height"]
ai_vision_dir = os.path.expanduser(neural_network_config["ai_vision_dir"])
exposurecompensation = neural_network_config["exposurecompensation"]
rotate_180 = "rotate-180" if neural_network_config["rotate_180"] else "none"

# ask for connector name
while True:
    connector = input("Please enter a connector: ")
    connector = os.path.splitext(connector)[0]
    
    if not os.path.exists(os.path.join(ai_vision_dir, "plugs",connector + '.csv')):
        print(f"Connector '{connector}' doesn't exist")
    elif not os.path.isdir(os.path.join(ai_vision_dir, "settings", connector)) or not os.listdir(os.path.join(ai_vision_dir, "settings", connector)):
        print(f"No configurations exist for connector {connector}, you can create them with the 'new_configuration.py' program")
    else:
        break

# ask for data directory name
while True:
    data_dir = input("Please enter a directory to save images to. (Can be a full directory or only a name): ")
    if not data_dir.startswith('/'):
        data_dir = os.path.join(ai_vision_dir, "data", data_dir)
    if not os.path.isdir(data_dir):
        print("Output directory doesn't exist, trying to create it now...")
        try:
            os.mkdir(data_dir)
        except exception as e:
            print("Could not create directory, the following error occured: ", str(e))
            continue
        else:
            break
    elif os.listdir(data_dir):
        print("Directory is not empty, new data will be added to existing directories")
        break
    else:
        break

# create input and output streams
input = jetson.utils.videoSource(video_source, argv=[f"--input-width={width}", f"--input-height={height}", f"--exposurecompensation={exposurecompensation}", f"--input-flip={rotate_180}"])
output = jetson.utils.videoOutput(video_sink, argv=['--headless'])

# capture and render first image
img = input.Capture()
output.Render(img)

left, top, right, bottom = load_connector(os.path.join(ai_vision_dir, "plugs", connector + '.csv'))

class TrainingWindow(QWidget):
    def __init__(self, capture_thread, connector):
        self.capture_thread = capture_thread
        super().__init__()
        self.setGeometry(50,50,1000,720)

        self.directory = data_dir
        self.housing = connector
        
        self.infoLabel = QLabel("Welcome to the Capture Training Data Program \n"
                                "Choose a plug and capture images to the presorted folders \n"
                                "For manual sorting, you can run the Image Sorter Program afterwards")

        # create videowidget
        self.videowidget = VideoWidget(parent=self)

        # create timer for auto capture
        self.autoTimer = QTimer()

        # create combobox containing the available connector configurations
        self.connectorHeadline = QLabel("Select a configuration")
        self.connectorSelector = QComboBox()
        for connector in sorted(os.listdir(os.path.join(ai_vision_dir, "settings", self.housing))):
            self.connectorSelector.addItem(connector)

        # create input for waiting time
        self.timeHeadline = QLabel("Waiting time for auto capture [ms]")
        self.timeInput = QLineEdit("300")
        self.timeInput.setValidator(QIntValidator(100,5000))

        # create input for empty chambers
        self.emptyHeadline = QLabel("Number of empty chambers")
        self.emptyInput = QLineEdit("0")
        self.emptyInput.setValidator(QIntValidator(0,1000))

        # create auto button
        self.autocheckbox = QCheckBox("Auto")

        #create start stop button
        self.start_stop_button = QPushButton("Capture")

        # create textfield for manual color
        # self.chamber_color = QLineEdit("")
        self.chamber_color = QComboBox()
        for color in ["bl", "br", "ge", "gr","gr_rt", "lg", "rs", "rt", "rt_ws", "sw", "vi"]:
            self.chamber_color.addItem(color)

        # create textfield for manual chamber number
        # self.chamber_number = QLineEdit("0")
        # self.chamber_number.setValidator(QIntValidator(0,len(left)-1))
        self.chamber_number = QComboBox()
        for number in range(0,130):
            self.chamber_number.addItem(str(number))
        
        # connector layout
        if not opt.manual_mode:
            self.connectorLayout = QVBoxLayout()
            self.connectorLayout.addWidget(self.connectorHeadline)
            self.connectorLayout.addWidget(self.connectorSelector)
        else:
            self.connectorLayout = QHBoxLayout()

            self.colorLayout = QVBoxLayout()
            self.colorLayout.addWidget(QLabel("Color"))
            self.colorLayout.addWidget(self.chamber_color)
            self.numberLayout = QVBoxLayout()
            self.numberLayout.addWidget(QLabel("Chamber"))
            self.numberLayout.addWidget(self.chamber_number)
            
            self.connectorLayout.addLayout(self.colorLayout)
            self.connectorLayout.addLayout(self.numberLayout)

        # waiting time layout
        self.waitingLayout = QVBoxLayout()
        self.waitingLayout.addWidget(self.timeHeadline)
        self.waitingLayout.addWidget(self.timeInput)

        # empty chamber layout
        self.emptyLayout = QVBoxLayout()
        self.emptyLayout.addWidget(self.emptyHeadline)
        self.emptyLayout.addWidget(self.emptyInput)

        # layout for time, auto and start
        self.horLayout = QHBoxLayout()
        self.horLayout.addLayout(self.connectorLayout)
        self.horLayout.addLayout(self.waitingLayout)
        self.horLayout.addLayout(self.emptyLayout)
        self.horLayout.addStretch()
        self.horLayout.addWidget(self.autocheckbox)
        self.horLayout.addWidget(self.start_stop_button)

        # connect events
        self.start_stop_button.clicked.connect(self.start_stop_pressed)
        self.autoTimer.timeout.connect(self.save_images)

        # create general layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.videowidget,stretch = 10)
        self.layout.addWidget(self.infoLabel, stretch = 1)
        self.layout.addLayout(self.horLayout, stretch = 1)

        self.setLayout(self.layout)
        self.setWindowTitle("Capture Training Data")

    def save_images(self):
        "crops and saves images"
        images = crop_images(left, top, right, bottom, img)

        # save chambers with wires
        for color, chamber in zip(self.colors, self.chambers):
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + ".jpg"
            if not os.path.isdir(os.path.join(self.directory, color)):
                print("directory ", color, " doesn't exist, trying to create now...")
                os.mkdir(os.path.join(self.directory, color))
            jetson.utils.cudaDeviceSynchronize()
            jetson.utils.saveImage(os.path.join(self.directory, color, str(chamber) + "_" + filename), images[chamber])
        
        # save empty chambers
        empty_chambers = random.sample(self.emptyList, int(self.emptyInput.text()))
        for chamber in empty_chambers:
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + ".jpg"
            if not os.path.isdir(os.path.join(self.directory, "empty")):
                print("directory empty doesn't exist, trying to create now...")
                os.mkdir(os.path.join(self.directory, "empty"))
            jetson.utils.cudaDeviceSynchronize()
            jetson.utils.saveImage(os.path.join(self.directory, "empty", str(chamber) + "_" + filename), images[chamber])

    def start_stop_pressed(self):
        # self.configuration = load_configuration(os.path.join(self.config_path, self.connectorSelector.currentText()))
        if not opt.manual_mode:
            with open(os.path.join(ai_vision_dir, "settings", self.housing, self.connectorSelector.currentText()), "r") as f:
                json_data = json.load(f)
            self.colors, self.chambers = create_expected_result(json_data)
        else:
            # self.colors = [self.chamber_color.text()]
            # self.chambers = [int(self.chamber_number.text())]
            self.colors = [self.chamber_color.currentText()]
            self.chambers = [int(self.chamber_number.currentText())]

        self.number_colors = len(self.colors)
        if int(self.emptyInput.text()) > len(left) - self.number_colors:
            self.emptyInput.setText(str(len(left) - self.number_colors))

        self.emptyInput.setValidator(QIntValidator(0,int(len(left) - self.number_colors)))

        self.emptyList = [item for item in range(len(left)) if not  item in self.chambers]
        
        if self.start_stop_button.text() == "Stop":
            self.autoTimer.stop()
            self.start_stop_button.setText("Capture")
        elif not self.autocheckbox.isChecked():
            self.save_images()
        else:
            self.start_stop_button.setText("Stop")
            self.autoTimer.start(int(self.timeInput.text()))

    def create_labels(self):
        folders = sorted([item for item in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, item))])
        with open(os.path.join(self.directory, "labels.txt"), 'w') as file:
            for element in folders:
                file.write("%s\n" % element)

    def create_occluded(self):
        if not os.path.exists(os.path.join(self.directory, "occluded")):
            os.mkdir(os.path.join(self.directory, "occluded"))
            
    def closeEvent(self, event):
        print("Closing Application")
        self.capture_thread.capturing_images = False
        # create occluded dir and labels.txt on close
        self.create_occluded()
        # self.create_labels()
        self.close()


class VideoWidget(QFrame):   
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)
        self.windowId = self.winId()

    def setup_pipeline(self):    
        self.pipeline = Gst.parse_launch("intervideosrc channel=v0 timeout=-1 ! xvimagesink")
        bus =  self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.enable_sync_message_emission()
        bus.connect('sync-message::element', self.on_sync_message)
  
    def on_sync_message(self, bus, msg):
        message_name = msg.get_structure().get_name()
        print(message_name)
        if message_name == 'prepare-window-handle':
            win_id = self.windowId
            assert win_id
            imagesink = msg.src
            imagesink.set_window_handle(win_id)
             
    def  start_pipeline(self):
        self.pipeline.set_state(Gst.State.PLAYING)


class ImageCapture(threading.Thread):
    def __init__(self):
        super(ImageCapture, self).__init__()
        self.capturing_images = True

    def run(self):
        while self.capturing_images:
            img = input.Capture()
            img2 = jetson.utils.cudaMemcpy(img)
            create_overlay(left, top, right, bottom, img2)
            output.Render(img2)
            time.sleep(0.04)


if __name__ == '__main__':
    GObject.threads_init()
    Gst.init(None)

    capture_thread = ImageCapture()
    capture_thread.start()

    app = QApplication([])
    window = TrainingWindow(capture_thread, connector)

    window.videowidget.setup_pipeline()
    window.videowidget.start_pipeline()

    window.show()
    sys.exit(app.exec_())
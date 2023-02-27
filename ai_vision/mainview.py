#!/usr/bin/python3

import sys
import os

import sys
import threading
import queue
import traceback
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstVideo

from functools import partial
from itertools import zip_longest
import logging

from neural_network import NeuralNework
from errorPicture import ErrorPicture
from krosy_handler import KrosyHandler
from krosy_server import KrosyServer
from krosy_client import KrosyClient
from logWidget import LogWidget
from helpers import *

# set high priority
# os.nice(-15)    # range -20 (high prio) to 19 (low prio)
dirname = os.path.dirname(os.path.realpath(__file__))

# get settings from gui_config.json
with open(os.path.join(dirname, "ressource", "gui_config.json")) as f:
    gui_config = json.load(f)
background = gui_config["background"]
label_background = gui_config["label_background"]
label_text = gui_config["label_text"]
label_font_size = gui_config["label_font_size"]
button_font_size = gui_config["button_font_size"]
cursor_active = gui_config["cursor_active"]
blink_time_ms = gui_config["blink_time_ms"]
interface_mode = gui_config["interface_mode"]
size_factor = gui_config["size_factor"]

# get network config
with open(os.path.join(dirname, "ressource", "network_config.json")) as f:
    network_config = json.load(f)
mode = network_config["mode"]

# get neural network config
with open(os.path.join(dirname, "ressource", "neural_network_config.json")) as f:
    neural_network_config = json.load(f)
ai_vision_dir = os.path.expanduser(neural_network_config["ai_vision_dir"])

# check mode parameter
if not mode in ["local", "krosy"]:
    print("please set mode to one of the following:\n"
    "\t'local' for using locally saved configurations\n"
    "\t'krosy' for using configurations received by krosy\n"
    f"you can change the mode in '{dirname}/ressource/network_config.json'")     #MAKE SURE THIS IS CORRECT PATH
    sys.exit(0)

# check the interface mode
if not interface_mode in ["auto", "bottom", "left", "right"]:
    print("please set interface mode to one of the following:\n"
    "\t'auto': detect screen rotation and choose between modes 'bottom' or 'right'\n"
    "\t'bottom': bar on the bottom\n"
    "\t'left': bar on the left\n"
    "\t'right': bar on the right\n"
    f"you can change the interface mode in '{dirname}/ressource/network_config.json'")     #MAKE SURE THIS IS CORRECT PATH
    sys.exit(0)

class FirstWindow(QWidget):   
    def __init__(self, app, nn_thread, error_queue, client, server):
        QMainWindow.__init__(self, None)
        self.nn_thread = nn_thread
        self.error_queue = error_queue
        self.client = client
        self.server = server

        self.setWindowTitle("AI Vision")
        self.setWindowIcon(QIcon(os.path.join(dirname,"ressource", "ai_vision.png")))
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        screen = app.primaryScreen()
        if not cursor_active:
            self.setCursor(Qt.BlankCursor)
        self.setStyleSheet(f"background-color: {background};")

        # setup video widget
        self.videowidget = VideoWidget(parent=self)

        # declare chamber object list for chambers with errors
        self.chamber_objects = []
 
        # get screen size
        width = screen.size().width()
        height = screen.size().height()

        # set picture size and stretch factor
        picture_size = size_factor * min(width, height)/10    
        self.nn_thread.picture_height = picture_size
        stretch_factor = height/(size_factor*min(width, height)/10)

        # automatically set the video rotation
        if width/height >= 1:
            self.videowidget.flip_method = "none"
        else:
            self.videowidget.flip_method = "clockwise"

        # calculate maximum number of pictures to show
        global interface_mode
        if interface_mode == "auto":
            if width/height >= 1:
                interface_mode = "bottom"
                number_of_images_to_show = int(0.9*(width//picture_size))
            else:
                interface_mode = "right"
                number_of_images_to_show = int(0.9*(height//picture_size))
        elif interface_mode == "bottom":
            number_of_images_to_show = int(0.9*(width//picture_size))
        elif interface_mode == "left" or interface_mode == "right":
            number_of_images_to_show = int(0.9*(height//picture_size))

        print("Höhe: ", height, " Breite: ", width, " Anzahl: ", number_of_images_to_show)

        # create labels and set images, create ErrorPicture Instances
        self.error_pitcture_objects = []
        self.wrongColorLabels = [QLabel() for i in range(number_of_images_to_show)]
        for i in range(number_of_images_to_show):
            self.wrongColorLabels[i].mouseReleaseEvent = partial(self.wrongColorClicked, i)
            self.error_pitcture_objects.append(ErrorPicture(i, picture_size, self.wrongColorLabels[i], os.path.join(dirname, "ressource", "color_config.json")))

        # create timer for images
        self.timer_ms = blink_time_ms
        self.image_timer_on = QTimer()
        self.image_timer_off = QTimer()
        self.image_timer_on.timeout.connect(self.blinkClickedImagesOn)
        self.image_timer_off.timeout.connect(self.blinkClickedImagesOff)
        self.image_timer_off.start(self.timer_ms)

        # thread for reading errors from NN
        self.stop_update_thread = False
        self.update_thread = threading.Thread(target=self.get_nn_result, daemon=True)
        self.update_thread.start()

        # create labels
        self.labelFont = QFont("Arial", label_font_size)

        self.TopInfoBox = QLabel("Spaceholder for Info (FPS, Errors, Status, etc")
        self.TopInfoBox.setStyleSheet(f"background-color: {label_background};"
                                        f"color: {label_text};")
        self.TopInfoBox.setFont(self.labelFont)
        self.TopInfoBox.setWordWrap(True)
        # self.logging = LogWidget(self.TopInfoBox)
        # self.nn_thread.logging = self.logging       # enable logging from nn thread
        self.logging = LogWidget()
        LogWidget.initialize(self.TopInfoBox)
        self.bottomLabel = QLabel("ERRORS")
        self.bottomLabel.setStyleSheet(f"background-color: {label_background};"
                                        f"color: {label_text};")
        self.bottomLabel.setFont(self.labelFont)
        self.bottomLabel.setAlignment(Qt.AlignCenter)

        # create control buttons
        self.buttonFont = QFont("Arial", button_font_size)

        self.loadButton = QPushButton("    LOAD    ")
        self.loadButton.setStyleSheet("background-color: #3d8bcd; color: white")
        self.loadButton.setFont(self.buttonFont)
        self.loadButton.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding)
        self.loadButton.clicked.connect(self.sendLocalData)

        # create close shortcut
        close_shortcut = QShortcut(QKeySequence('Ctrl+Q'), self)
        close_shortcut.activated.connect(self.closeApp)

        # create combo box
        self.configurationComboBox = QComboBox()
        self.configurationComboBox.setStyleSheet("background-color: #3d8bcd; color: white")
        self.configurationComboBox.setFont(self.buttonFont)
        # self.configurationComboBox.setEditable(True)
        # self.configurationComboBox.lineEdit().setAlignment(Qt.AlignCenter)
        # self.configurationComboBox.lineEdit().setReadOnly(True)
        self.configurationComboBox.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding)

        self.plug_paths = []
        for plugs in os.listdir(os.path.join(ai_vision_dir, "settings")):
            if os.path.isdir(os.path.join(ai_vision_dir, "settings",plugs)):
                for json_file in os.listdir(os.path.join(ai_vision_dir, "settings", plugs)):
                    self.configurationComboBox.addItem(json_file)
                    self.plug_paths.append(os.path.join(ai_vision_dir, "settings", plugs, json_file))

        # top strip layout
        topLayout = QHBoxLayout()
        topLayout.setSpacing(0)    
        topLayout.addWidget(self.TopInfoBox, stretch=10)
        if mode == "local":
            topLayout.addWidget(self.configurationComboBox, stretch=1)
            topLayout.addWidget(self.loadButton, stretch=1)

        # layout containing the error pictures
        if interface_mode == "bottom":
            picturesLayout = QHBoxLayout()
        else:
            picturesLayout = QVBoxLayout()
        picturesLayout.setSpacing(0)
        for wrongColorLabel in self.wrongColorLabels:
            picturesLayout.addWidget(wrongColorLabel)
        filler = QLabel()
        filler.setStyleSheet(f"background-color: {label_background};")
        filler.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        picturesLayout.addWidget(filler)

        # middle strip layout 
        middleLayout = QHBoxLayout()
        middleLayout.setSpacing(0)
        if interface_mode == "left":
            middleLayout.addLayout(picturesLayout)
        middleLayout.addWidget(self.videowidget, stretch=8)
        if interface_mode == "right":
            middleLayout.addLayout(picturesLayout)

        # complete layout
        layout = QVBoxLayout()
        layout.setSpacing(0)                        # spacing between contents
        layout.setContentsMargins(0,0,0,0)          # outer border size
        layout.addLayout(topLayout, stretch=1)
        layout.addLayout(middleLayout, stretch=stretch_factor)
        if interface_mode == "bottom":
            layout.addLayout(picturesLayout)
        self.setLayout(layout)

    def closeApp(self):
        """stop all threads and close App"""
        self.stop_update_thread = True
        self.nn_thread.closeNN()
        self.nn_thread.closeDetection()
        self.server.stop()
        self.close()

    def stopNeuralNetwork(self):
        """stop detection"""
        if nn_thread.nn_is_running:
            print("Closing detection")
            self.logging.set_info(1, "Closing detection")
            self.nn_thread.closeDetection()
        else:
            print("Detection isn't running, nothing to stop")
            self.logging.set_info(1, "Detection isn't running, nothing to stop")

    def sendLocalData(self):
        self.client.send_data(self.plug_paths[self.configurationComboBox.currentIndex()])

    def wrongColorClicked(self, number, event):
        """sets clicked ErrorPicture object to clicked, renders images on screen"""
        QApplication.processEvents()
        QApplication.processEvents()
        if not self.error_pitcture_objects[number].get_clickable():
            number = ErrorPicture.active_picture
        for idx, error_picture in enumerate(self.error_pitcture_objects):
            if idx == number:
                error_picture.set_clicked()
            else:
                error_picture.set_unclicked()

    def blinkClickedImagesOn(self):
        """changes clicked image to active"""
        self.image_timer_on.stop()
        try:
            if nn_thread.detection_running:
                for error_picture, chamber in zip(self.error_pitcture_objects, self.chamber_objects):
                    if error_picture.get_clicked():
                        error_picture.set_active()
                        chamber.set_active(True)
                        self.logging.set_info_nolog(0, f"Result: {chamber.last_result}, Confidence: {chamber.last_confidence}")
                    else:
                        error_picture.set_inactive()
                        chamber.set_active(False) 
                    error_picture.get_image()
        except Exception:
            print("-----------------------------------------------")
            traceback.print_exc()
            print("-----------------------------------------------")
        QApplication.processEvents()
        QApplication.processEvents()
        self.image_timer_off.start(self.timer_ms)
        # print(ErrorPicture.active_picture)

    def blinkClickedImagesOff(self):
        """changes all images to inactive"""
        self.image_timer_off.stop()
        try:
            if nn_thread.detection_running:
                for error_picture, chamber in zip(self.error_pitcture_objects, self.chamber_objects):
                    error_picture.set_inactive()
                    error_picture.get_image()
                    chamber.set_active(False)
        except Exception:
            print("-----------------------------------------------")
            traceback.print_exc()
            print("-----------------------------------------------")
        QApplication.processEvents()
        self.image_timer_on.start(self.timer_ms)

    def get_nn_result(self):
        while not self.stop_update_thread:
            self.chamber_objects = self.error_queue.get()
            # if parallel chambers > max number of pictures, truncate list
            if len(self.chamber_objects) > len(self.error_pitcture_objects):
                del self.chamber_objects[len(self.error_pitcture_objects):]
            for chamber, error_picture in zip_longest(self.chamber_objects, self.error_pitcture_objects):
                if chamber != None:
                    error_picture.last_confidence = chamber.last_confidence     # debugging
                    error_picture.last_result = chamber.last_result             # debugging
                    error_picture.set_images(*chamber.get_images())
                else:
                    error_picture.set_reset()
                error_picture.get_image()
                QApplication.processEvents()
            
            if ErrorPicture.active_picture == -1 and len(self.chamber_objects) > 0:
                self.wrongColorClicked(0, self.event)
            elif ErrorPicture.active_picture >= len(self.chamber_objects):
                self.wrongColorClicked(len(self.chamber_objects)-1, self.event)    
            else:
                self.wrongColorClicked(ErrorPicture.active_picture, self.event)


class VideoWidget(QWidget):   
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)
        self.flip_method = "none"
        self.windowId = self.winId()
        self.setStyleSheet(f"background-color: {background};")

    def setup_pipeline(self):
        self.pipeline = Gst.parse_launch(f"intervideosrc channel=v0 timeout=-1 ! videoflip method={self.flip_method} ! xvimagesink") #rotate-180
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


if __name__ == "__main__":
    GObject.threads_init()
    Gst.init(None)

    app = QApplication([])

    # start nerual network
    error_queue = queue.Queue(maxsize=-1)
    nn_thread = NeuralNework(error_queue)

    # start webserver for krosy
    server = KrosyServer(nn_thread, os.path.join(ai_vision_dir, "krosy"))
    client = KrosyClient()

    
    # setup pipeline for video output
    window = FirstWindow(app, nn_thread, error_queue, client, server)
    window.videowidget.setup_pipeline()
    window.videowidget.start_pipeline()
 
    server.start()
    nn_thread.start()
    window.showFullScreen()
    sys.exit(app.exec_())
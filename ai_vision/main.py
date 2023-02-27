#!/usr/bin/python3

import jetson.inference
import jetson.utils

import argparse
import sys
import time
import collections


from itertools import count
import sys
import threading
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstVideo

from mainview import FirstWindow

from neural_network import NeuralNework


if __name__ == "__main__":
    GObject.threads_init()
    Gst.init(None)

    app = QApplication([])

    # start nerual network
    nn_thread = NeuralNework()
    
    # setup pipeline for video output
    window = FirstWindow(app, nn_thread)
    window.videowidget.setup_pipeline()
    window.videowidget.start_pipeline()

    nn_thread.start()
    window.show()
    sys.exit(app.exec_())
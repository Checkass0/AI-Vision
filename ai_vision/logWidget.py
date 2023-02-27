import logging
import os

dirname = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=os.path.join(dirname, "ressource", "logfile.log"), format='%(asctime)s | %(levelname)s: %(message)s', level=logging.DEBUG, filemode='w')

class LogWidget():
    """Enable logging to logfile and LogWidget"""
    
    @staticmethod
    def initialize(infoBox):
        LogWidget.log_text = ["",""]
        LogWidget.infoBox = infoBox
        
    @staticmethod
    def set_debug(text: str):
        logging.debug(text)

    @staticmethod
    def set_info_nolog(logID: int, text: str):
        try:
            LogWidget.log_text[logID] = text
            LogWidget.infoBox.setText(' | '.join(LogWidget.log_text))
        except: pass
    
    @staticmethod
    def set_info(logID: int, text: str):
        logging.info(text)
        try:
            LogWidget.log_text[logID] = text
            LogWidget.infoBox.setText(' | '.join(LogWidget.log_text))
        except: pass

    @staticmethod
    def set_warning(logID: int, text: str):
        logging.warning(text)
        try:
            LogWidget.log_text[logID] = text
            LogWidget.infoBox.setText(' | '.join(LogWidget.log_text))
        except: pass

    @staticmethod
    def set_error(logID: int, text: str):
        logging.error(text)
        try:
            LogWidget.log_text[logID] = text
            LogWidget.infoBox.setText(' | '.join(LogWidget.log_text))
        except: pass

    @staticmethod
    def remove(logID: int):
        try:
            LogWidget.log_text[logID] = ""
            LogWidget.infoBox.setText(' | '.join(LogWidget.log_text))
        except: pass


# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# import gi
# gi.require_version('Gst', '1.0')
# gi.require_version('GstVideo', '1.0')
# from gi.repository import Gst, GObject, GstVideo
# import sys
# import time


# class FirstWindow(QWidget):   
#     def __init__(self):
#         QMainWindow.__init__(self, None)

#         self.setWindowTitle("AI Vision")
#         self.setWindowIcon(QIcon(os.path.join(dirname,"ressource", "icon.png")))
#         self.setAttribute(Qt.WA_AcceptTouchEvents, True)


#         self.TopInfoBox = QLabel("Spaceholder for Info (FPS, Errors, Status, etc")
#         LogWidget.initialize(self.TopInfoBox)

#         layout = QVBoxLayout()
#         layout.addWidget(self.TopInfoBox)
#         self.setLayout(layout)


# if __name__ == "__main__":
#     app = QApplication([])

#     LogWidget.set_info(0, "vor init")
    
#     # setup pipeline for video output
#     window = FirstWindow()

#     window.show()
#     LogWidget.set_info(0, "test nummer 1")

#     log1 = LogWidget()
#     log2 = LogWidget()

#     log1.set_info(1, "log1")
#     log2.set_info(1, "log2")
#     sys.exit(app.exec_())
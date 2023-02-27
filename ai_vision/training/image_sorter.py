import argparse
import sys
import os
import shutil
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import json

# parse the command line
parser = argparse.ArgumentParser(description = "opens all images in subfolders and creates CLI to move them into different folder",
                                formatter_class=argparse.RawTextHelpFormatter)

# parser.add_argument("input_dir", type=str, default="", help="directory containing directories of images")

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
    network_config = json.load(f)
ai_vision_dir = os.path.expanduser(network_config["ai_vision_dir"])

class ImageWindow(QWidget):
    def __init__(self, input_dir):
        self.input_directory = input_dir
        super().__init__()

        # initialize widgets
        self.pictureLabel = QLabel()
        self.dir_label = QLabel()
        self.next_dir_button = QPushButton("Next Directory")
        self.undo_button = QPushButton("Undo")
        self.text_input = QLineEdit()
        self.text_output = QLabel("Welcome to the sorting helper.\n"
                                    "Type the name of the folder or hit Enter to leave Image in current Folder\n"
                                    "Shortcuts: \t'o' for occluded \n"
                                    "\t\t'e' for empty\n"
                                    "\t\t'd' or 'delete' to delete image\n"
                                    "\t\t'h' or 'help' for help")
        self.text_output.setWordWrap(True)
        self.diashow_checkbox = QCheckBox("Diashow")
        self.msgBox = QMessageBox()
        self.msgBox.setModal(True)
        self.msgBox.setIcon(QMessageBox.Information)
        self.msgBox.setWindowTitle("INFO")
        self.msgBox.setStandardButtons(QMessageBox.Ok)

        self.bottomDirLayout = QVBoxLayout()
        self.bottomDirLayout.addWidget(QLabel("Current Directory:"))
        self.bottomDirLayout.addWidget(self.dir_label)
        self.bottomDirLayout.addWidget(self.next_dir_button)
        self.next_dir_button.clicked.connect(self.set_next_directory)
        self.bottomDirLayout.addWidget(self.undo_button)
        self.undo_button.clicked.connect(self.undo)

        self.bottomHorLayout = QHBoxLayout()
        self.bottomHorLayout.addWidget(self.text_input)
        self.bottomHorLayout.addWidget(self.diashow_checkbox)
        # self.diashow_checkbox.clicked.connect(self.start_diashow)

        self.bottomInputLayout = QVBoxLayout()
        self.bottomInputLayout.addWidget(QLabel("Move Image to:"))
        self.bottomInputLayout.addLayout(self.bottomHorLayout)
        self.text_input.returnPressed.connect(self.get_user_input)

        self.bottomLayout = QHBoxLayout()
        self.bottomLayout.addLayout(self.bottomDirLayout, stretch=1)
        self.bottomLayout.addSpacing(10)
        self.bottomLayout.addLayout(self.bottomInputLayout, stretch=10)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.text_output, 1)
        self.layout.addWidget(self.pictureLabel, 10)
        self.layout.addLayout(self.bottomLayout, 1)
        
        self.setLayout(self.layout)
        # self.setGeometry(50,50,320,200)
        self.setWindowTitle("Image Sorter")
        self.show()

        self.text_input.setFocus()

        # timer for diashow
        self.diashowTimer = QTimer()
        self.diashowTimer.timeout.connect(self.start_diashow)
        self.diashowTimer.start(100)

        # create buffer for undo operations
        # self.undo_button.setDisabled(True)
        self.number_of_current_image = 0
        self.number_of_current_dir = -1
        self.buffer = [[]]

        # # check if file path for data dir was specified
        # if opt.input_dir.startswith('/'):
        #     self.input_directory = opt.input_dir
        # else:
        #     self.input_directory = os.path.join(ai_vision_dir, "data", opt.input_dir)

        # # check if it exists
        # if not os.path.isdir(self.input_directory):
        #     print("input directory doesn't exist")
        #     self.text_output.setText("input directory doesn't exist")
        #     self.closeApp()

        # create list of directories
        self.sub_directories = sorted([item for item in os.listdir(self.input_directory) if os.path.isdir(os.path.join(self.input_directory, item))])

        # set shortcuts for empty and occluded
        self.empty_shortcut = "empty" in self.sub_directories
        self.occluded_shortcut = "occluded" in self.sub_directories

        if any(item in ["o","e","h","help","d","delete","u", "undo"] for item in self.sub_directories):
            print("directories with the nammes 'o','e','h','help','d','delete','u','undo' are not allowed, shutting down now...")
            sys.exit(0)

        
        if len(self.sub_directories) == 0:
            print("input directory is empty")
            self.text_output.setText("input directory is empty")
            self.closeApp()
        else:
            self.set_image()

    def set_next_directory(self):
        self.number_of_current_dir += 1
        self.set_directory()

    def set_directory(self):
        self.text_input.setFocus()
        if self.number_of_current_dir >= len(self.sub_directories):
            print("no directory left, closing app...")
            self.text_output.setText("no directory left, closing app...")
            self.closeApp()
        else:
            self.current_directory = self.sub_directories[self.number_of_current_dir]
            self.dir_label.setText(self.current_directory)
            print("New Directory opened: " + self.current_directory)
            if self.number_of_current_dir != 0:
                self.msgBox.about(self, "Info", "New Directory opened: " + self.current_directory)
            images = os.listdir(os.path.join(self.input_directory, self.current_directory))
            images.sort(key=lambda x: int(x.split("_")[0]))
            self.number_of_images = len(images)
            self.number_of_current_image = 0
            image_dirs = [os.path.join(self.input_directory, self.current_directory, image_name) for image_name in images]
            self.buffer = [images, image_dirs]
            self.set_image()

    def set_image(self):
        if self.number_of_current_image >= len(self.buffer[0]):
            self.number_of_current_dir += 1
            self.set_directory()
        else:
            self.dir_label.setText(self.current_directory + " [" + str(self.number_of_current_image) + "/" + str(self.number_of_images - 1) + " ]")
            # self.im = QPixmap(os.path.join(self.input_directory, self.current_directory, self.buffer[0][self.number_of_current_image])).scaledToHeight(300,Qt.SmoothTransformation)
            self.im = QPixmap(os.path.join(self.input_directory, self.current_directory, self.buffer[0][self.number_of_current_image]))
            self.pictureLabel.setPixmap(self.im,)
            self.pictureLabel.setScaledContents(True)
            self.pictureLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def get_user_input(self):
        user_input = self.text_input.text()
        self.text_input.clear()

        if user_input == "":
            print("Image wasn't moved")
            self.text_output.setText("Image wasn't moved")
            self.buffer[1][self.number_of_current_image] = os.path.join(self.input_directory, self.current_directory, self.buffer[0][self.number_of_current_image])
            self.number_of_current_image += 1
            self.set_image()

        elif user_input == "delete" or user_input == "d":
            os.remove(os.path.join(self.input_directory, self.current_directory, self.buffer[0][self.number_of_current_image]))
            print("Image was deleted")
            self.text_output.setText("Image was deleted")
            # deleted images cant be recovered
            self.number_of_current_image += 1
            self.set_image()

        elif user_input == "e" and self.empty_shortcut:
            newdir = os.path.join(self.input_directory, "empty")
            shutil.move(os.path.join(self.input_directory, self.current_directory, self.buffer[0][self.number_of_current_image]), os.path.join(newdir, self.buffer[0][self.number_of_current_image]))
            print("Image was moved to: ", newdir)
            self.text_output.setText("Image was moved to: '" + str(newdir) + "'")
            self.buffer[1][self.number_of_current_image] = os.path.join(newdir, self.buffer[0][self.number_of_current_image])
            self.number_of_current_image += 1
            self.set_image()

        elif user_input == "o" and self.occluded_shortcut:
            newdir = os.path.join(self.input_directory, "occluded")
            shutil.move(os.path.join(self.input_directory, self.current_directory, self.buffer[0][self.number_of_current_image]), os.path.join(newdir, self.buffer[0][self.number_of_current_image]))
            print("Image was moved to: ", newdir)
            self.text_output.setText("Image was moved to: '" + str(newdir) + "'")
            self.buffer[1][self.number_of_current_image] = os.path.join(newdir, self.buffer[0][self.number_of_current_image])
            self.number_of_current_image += 1
            self.set_image()

        elif user_input == "h" or user_input == "help":
            self.text_output.setText("Welcome to the sorting helper. Type the name of the folder or hit Enter to leave Image in current Folder\n"
                                    "Shortcuts: \t'o' for occluded \n"
                                    "\t\t'e' for empty\n"
                                    "\t\t'd' or 'delete' to delete image\n"
                                    "\t\t'h' or 'help' for help")

        elif user_input in self.sub_directories:
            newdir = os.path.join(self.input_directory, user_input)
            shutil.move(os.path.join(self.input_directory, self.current_directory, self.buffer[0][self.number_of_current_image]), os.path.join(newdir, self.buffer[0][self.number_of_current_image]))
            print("Image was moved to: ", newdir)
            self.text_output.setText("Image was moved to: '" + str(newdir) + "'")
            self.buffer[1][self.number_of_current_image] = os.path.join(newdir, self.buffer[0][self.number_of_current_image])
            self.number_of_current_image += 1
            self.set_image()
        elif user_input == "u" or user_input == "undo":
            self.undo()

        else: 
            print("Input '", user_input,"' was wrong, try again...")
            self.text_output.setText("Input '" + user_input + "' was wrong, try again...")

    def undo(self):
        self.text_input.setFocus()
        if self.number_of_current_image == 0:
            # TODO: Go back through sub-directories. Saving of previous buffers needed
            # if self.number_of_current_dir > 0:
            #     self.number_of_current_dir -= 1
            #     self.current_directory = self.sub_directories[self.number_of_current_dir]
            #     self.set_directory()
            #     self.number_of_current_image = len(self.buffer[0])
            #     self.undo()
            # else:
            #     print("nothing to undo")
            #     self.text_output.setText("Nothing to undo")
            print("Can't go back further")
            self.text_output.setText("Can't go back further")
        else:
            self.number_of_current_image -= 1
            self.dir_label.setText(self.current_directory + " [" + str(self.number_of_current_image) + "/" + str(self.number_of_images) + " ]")
            origin = self.buffer[1][self.number_of_current_image]
            destination = os.path.join(self.input_directory, self.current_directory, self.buffer[0][self.number_of_current_image])
            shutil.move(origin, destination)
            # self.text_output.setText("Undo: moved image " + origin + "\nback to " + destination)
            self.text_output.setText("Undo: moved image " + origin + "\nback to origin")
            self.set_image()

    def start_diashow(self):
        if self.diashow_checkbox.isChecked():
            self.text_output.setText("Started Diashow, going through images now...")
            self.number_of_current_image += 1
            self.set_image()

    def closeApp(self):
        self.close()

if __name__ == '__main__':

    # ask for connector name
    while True:
        input_dir = input("Please enter a directory: ")
        if not input_dir.startswith('/'):
            input_dir = os.path.join(ai_vision_dir, "data", input_dir)

        if not os.path.isdir(input_dir):
            print(f"Directory '{input_dir}' doesn't exist")
        elif not os.listdir(input_dir):
            print(f"Directory '{input_dir}' is empty")
        else:
            break

    app = QApplication([])
    window = ImageWindow(input_dir)
    sys.exit(app.exec_())




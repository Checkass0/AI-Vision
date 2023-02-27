#!/usr/bin/python3

import jetson.inference
import jetson.utils

import os
import threading
import json

from helpers import *
from chamber import Chamber
import queue

from logWidget import LogWidget

dirname = os.path.dirname(os.path.realpath(__file__))

# load settings from neural network config file
with open(os.path.join(dirname, "ressource", "neural_network_config.json")) as f:
    neural_network_config = json.load(f)
video_source = neural_network_config["video_source"]
video_sink = neural_network_config["video_sink"]
input_blob = neural_network_config["input_blob"]
output_blob = neural_network_config["output_blob"]
width = neural_network_config["width"]
height = neural_network_config["height"]
threshold = neural_network_config["threshold"]
buffer_size = neural_network_config["buffer_size"]
parallel_detections = neural_network_config["parallel_detections"]
ai_vision_dir = os.path.expanduser(neural_network_config["ai_vision_dir"])
model = os.path.join(ai_vision_dir, "models", neural_network_config["model"], "resnet18.onnx")
labels = os.path.join(ai_vision_dir, "models", neural_network_config["model"], "labels.txt")
exposurecompensation = neural_network_config["exposurecompensation"]
rotate_180 = "rotate-180" if neural_network_config["rotate_180"] else "none"

# load the recognition network
net = jetson.inference.imageNet(argv=[f"--model={model}", f"--labels={labels}", f"--input_blob={input_blob}", f"--output_blob={output_blob}"])

# create video sources & outputs
input = jetson.utils.videoSource(video_source, argv=[f"--input-width={width}", f"--input-height={height}", f"--exposurecompensation={exposurecompensation}", f"--input-flip={rotate_180}", "--log-level=error"])
output = jetson.utils.videoOutput(video_sink, argv=["--headless"]) 

# run classification once because of bug in jetson inference code
test_img = jetson.utils.loadImage(os.path.join(dirname, "ressource", "default.jpg"))
class_id, confidence = net.Classify(test_img)
print("First run of neural network gave: ", net.GetClassDesc(class_id), confidence)

image_queue = queue.Queue(maxsize=1)

class NeuralNework(threading.Thread):
    def __init__(self, error_queue):
        super(NeuralNework, self).__init__()
        # global variables to control thread from main
        self.stop_neural_network = False    # stops nn if true
        self.force_detection_stop = False   # stops detection if true
        self.detection_running = False      # True when detection is running
        self.nn_is_running = False
        self.fps = 0
        self.picture_height = 300       # size of the error pictures in pixels, gets overwritten in mainview
        self.error_queue = error_queue
        self.result = None
        self.expected_result = None
        self.start_detection_event = threading.Event()      # gets set from main with nn_thread.start_detection_event.set()
        self.logging = LogWidget()

        # create video sources & outputs
        self.input = input
        self.output = output

    def run(self):
        # create threads for input and nn
        input_thread = threading.Thread(target=self.inputThread)

        # start input thread
        input_thread.start()

        while not self.stop_neural_network:
            # wait for scan event
            self.nn_is_running = False
            print("waiting for Detection start")
            neural_network_thread = threading.Thread(target=self.neuralNetworkThread)
            self.start_detection_event.wait()
            if not self.stop_neural_network:
                self.logging.set_info(1, "Starting Detection")
                self.nn_is_running = True
                neural_network_thread.start()
                neural_network_thread.join()
            else:
                input_thread.join()     # wait for input thread to close
        print("NN Thread was closed")

    def neuralNetworkThread(self):
        """start detection"""
        # reset flags
        self.force_detection_stop = False
        self.start_detection_event.clear()

        # get paths from json file
        with open(os.path.join(ai_vision_dir, "krosy", "data.json"), "r") as f:
            json_data = json.load(f)
        plug = os.path.join(ai_vision_dir, "plugs", json_data["plug"] + ".csv")

        # load connector from csv file
        [left,top,right,bottom] = load_connector(plug)

        # new: changed json format
        expected_results, chambers = create_expected_result(json_data)

        # load all avialable classes
        all_available_results = [net.GetClassDesc(idx) for idx in range(0,net.GetNumClasses())]

        # set number of parallel chambers
        Chamber.chunk_size = parallel_detections
        Chamber.height = self.picture_height
        Chamber.dirname = dirname

        # create chamber objects
        self.chamber_objects = []
        for idx, (expected_result, number) in enumerate(zip(expected_results, chambers)):
            self.chamber_objects.append(Chamber(idx, number, (left[number],top[number],right[number],bottom[number]), expected_result, all_available_results, threshold=threshold, length=buffer_size))

        # put first items in queue
        self.error_queue.put(Chamber.parallel_chambers)

        # set the start detection flag
        self.detection_running = True

        # classify multiple chambers in parallel
        while True:
            chambers = Chamber.parallel_chambers
            img = image_queue.get()
            for chamber in chambers:
                # crop the image
                jetson.utils.cudaDeviceSynchronize()
                cropped_img = crop_image(*chamber.get_position(), img)

                # TODO: Event queue, der im iunput thread mit bildern gefüllt wird.
                # Dieser wird nach while True gelesen, bzw auf event gewartet, falls notwendig.
                # Danach alle kammern ausschneiden und klassifizieren

                # classify the image
                jetson.utils.cudaDeviceSynchronize()
                class_id, confidence = net.Classify(cropped_img)
                class_desc = net.GetClassDesc(class_id)
                class_desc = class_desc.rstrip()        # carriage return from labels.txt file has to be removed

                # update fps
                self.fps = net.GetNetworkFPS()

                # add result to chamber
                chamber.set_result(class_desc, confidence)
                # print("Chamber ", chamber.get_chamber_number(), " was detected as: ", class_desc, ", confidence: ", confidence)

                # set chamber correct
                if chamber.get_correct():
                    # self.logging.set_info(1, f"Chamber {chamber.get_expected_result()} is correct")
                    # chamber.set_active_lock()
                    # chamber.set_correct()
                    self.error_queue.put(Chamber.parallel_chambers)

            if Chamber.parallel_chambers == []:
                print("All chambers are correct", Chamber.parallel_chambers)
                self.logging.set_info(1, "All chambers are correct")
                break
            if self.force_detection_stop:
                self.logging.set_info(1, "Detection was closed")
                for chamber in self.chamber_objects:
                    chamber.set_active_lock()
                    chamber.set_correct()
                break
            if self.stop_neural_network:
                print("Stopping Neural Network...")
                self.logging.set_info(1, "Detection was closed")
                break

        # reset flags, clear queue, reset chambers
        self.detection_running = False
        self.nn_is_running = False
        self.error_queue.put([None])
        self.chamber_objects = []

        print("Detection was closed")
     
    def inputThread(self):
        """capture images and create output stream"""
        while True:
            jetson.utils.cudaDeviceSynchronize()
            try:
                img = self.input.Capture()
            except Exception as e:
                self.logging.set_error(1, str(e))
            jetson.utils.cudaDeviceSynchronize()
            img2 = jetson.utils.cudaMemcpy(img)
            if image_queue.empty():
                image_queue.put_nowait(img2)
        
            try:
                jetson.utils.cudaDeviceSynchronize()
                for chamber in self.chamber_objects:
                    chamber.set_overlay(img)
            # replace with specific exception later
            except Exception as e:
                # print("an exception occured: ", e)
                pass
            self.output.Render(img)

            if self.stop_neural_network:
                break
        self.input.Close()
        print("Input thread was closed")

    def closeNN(self):
        print("closing neural network...")
        self.stop_neural_network = True
        self.start_detection_event.set()

    def closeDetection(self):
        print("closing detection...")
        self.force_detection_stop = True


if __name__ == "__main__":
    
    nn_thread = NeuralNework()
    nn_thread.start()
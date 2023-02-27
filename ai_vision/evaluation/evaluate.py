from random import random
import jetson.inference
import jetson.utils

import argparse
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap


# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str, help="directory to classify")
parser.add_argument("result", type=str, help="expected result (if the network detects a different color, the confidence will be 0)")
parser.add_argument("--load", type=bool, default=False, help="loads the test results from '<directory>/<result>.txt' if set to True")
parser.add_argument("--width", type=int, default=26, help="number of chambers horizontally")
parser.add_argument("--height", type=int, default=5, help="number of chambers vertically")
parser.add_argument("--exclude_occluded", type=bool, default=False, help="Set to true if you want to exclude occluded results")
parser.add_argument("--threshold", type=float, default=0, help="Set a threshold if you wish to exclude results under threshold")
parser.add_argument("--show", type=bool, default=False, help="Set Set True if you wish to show plot")
parser.add_argument("--constant_confidence", type=bool, default=False, help="Set True if you wish to set confidence to 100 if correct")

opt = parser.parse_args()

dirname = os.path.dirname(os.path.realpath(__file__))
dirname = os.path.dirname(dirname)

# load settings from neural network config file
with open(os.path.join(dirname, "ressource", "neural_network_config.json")) as f:
    neural_network_config = json.load(f)
input_blob = neural_network_config["input_blob"]
output_blob = neural_network_config["output_blob"]
ai_vision_dir = os.path.expanduser(neural_network_config["ai_vision_dir"])
model = os.path.join(ai_vision_dir, "models", neural_network_config["model"], "resnet18.onnx")
labels = os.path.join(ai_vision_dir, "models", neural_network_config["model"], "labels.txt")

# check if file path for data dir was specified
if opt.directory.startswith('/'):
    data_dir = opt.directory
else:
    data_dir = os.path.join(ai_vision_dir, "data", opt.directory)

if not os.path.exists(data_dir):
    print("directory doesn't exist")
    sys.exit(0)

# create distinct names for different thresholds, or excluding occluded results
if opt.exclude_occluded:
    occluded_name = "exo"
else:
    occluded_name = "ino"

if opt.constant_confidence:
    confidence_name = "cc"
else:
    confidence_name = "vc"

threshold_name = f"t{opt.threshold}".replace(".","-")
resultname = f"{opt.result}_{occluded_name}_{threshold_name}_{confidence_name}"

print(f"Executing program with color {opt.result}, excluding occluded pictures={opt.exclude_occluded}, threshold={opt.threshold}, constant confidence={opt.constant_confidence}")

width = opt.width
height = opt.height
chamber_nr = -1
conf = 0
number_per_chamber = 0
confidenceList = []
resultList = [0] * (height*width)
chamberList = []
all_descriptions = []
desc_list = []
all_results = [[]]
number_correct = 0
counter = 0

if opt.load:
    with open(os.path.join(os.path.dirname(data_dir), f"{resultname}.txt"), 'rb') as f:
        result = np.load(f)
else:
# load the recognition network
    net = jetson.inference.imageNet(argv=[f"--model={model}", f"--labels={labels}", f"--input_blob={input_blob}", f"--output_blob={output_blob}", "--log-level=error"])

    images = os.listdir(data_dir)
    images.sort(key=lambda x: int(x.split("_")[0]))

    
    for image in images:
        img = jetson.utils.loadImage(os.path.join(data_dir, image))

        # classify the image
        class_idx, confidence = net.Classify(img)

        # find the object description
        class_desc = net.GetClassDesc(class_idx)

        # append description to all results
        if not class_desc in all_descriptions:
            all_descriptions.append(class_desc)

        # check if occluded or empty
        if (class_desc == 'occluded' or class_desc == 'empty') and opt.exclude_occluded:
            pass
        # check for confidence
        elif confidence < opt.threshold:
            pass
        # first iteration
        elif chamber_nr == -1:
            if class_desc != opt.result:
                confidence = 0
            elif opt.constant_confidence:
                confidence = 1
            chamber_nr = int(image.split("_")[0])
            confidenceList.append(confidence)
        # finish list
        elif chamber_nr != int(image.split("_")[0]):
            if class_desc != opt.result:
                confidence = 0
            else:
                counter +=1
                if opt.constant_confidence:
                    confidence = 1
            resultList[chamber_nr] = int(100*sum(confidenceList)/len(confidenceList))
            chamberList.append(chamber_nr)
            # number_correct += len(confidenceList)
            print(f"the chamber {chamber_nr} was recognized with {resultList[chamber_nr]}% confidence from {len(confidenceList)} pictures (total: {len(desc_list)}), {Counter(desc_list)}")
            # set new chamber nr and confidence
            chamber_nr = int(image.split("_")[0])
            confidenceList = [confidence]
            desc_list = [class_desc]
        else:
            if class_desc != opt.result:
                confidence = 0
            else:
                counter +=1
                if opt.constant_confidence:
                    confidence = 1
            confidenceList.append(confidence)
        
        # append all results to desc
        desc_list.append(class_desc)

        # print out the result
        # print(f"chamber {chamber_nr}, image {image}, recognized as '{class_desc}', conficence: {confidence}%")

    

    resultList[chamber_nr] = int(100*sum(confidenceList)/len(confidenceList))
    chamberList.append(chamber_nr)
    # number_correct += len(confidenceList)
    # print last chamber
    print(f"the chamber {chamber_nr} was recognized with {resultList[chamber_nr]}% confidence from {len(confidenceList)} pictures (total: {len(desc_list)}), {Counter(desc_list)}")

    print(f"correct: {counter}")
    chamberList = [str(i) for i in chamberList]
    resultList = [i for i in resultList]

    result = np.reshape(resultList, (height, width))

    # save result to file
    with open(os.path.join(os.path.dirname(data_dir), f"{resultname}.txt"), 'wb') as f:
        np.save(f, result)


# plot größe
plt.rcParams["figure.figsize"] = (12,3)

# limit the viridis colormap
interval = np.linspace(0, 0.87)
colors = plt.cm.viridis(interval)
colormap = LinearSegmentedColormap.from_list('name', colors)

plt.figure()
ax = plt.gca()
ax.set_title(f"Confidence of color '{opt.result}' by chambers in %")
ax.set_xticks([])
ax.set_yticks([])
im = ax.imshow(result, cmap=colormap, vmin=0, vmax=100)
for i in range(width):
    for j in range(height):
        text = ax.text(i,j, result[j,i], ha="center", va="center", color="w", size=10)
    
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
   
plt.colorbar(im, cax=cax)
plt.subplots_adjust(left=0.01, right=0.97)
plt.savefig(os.path.join(os.path.dirname(data_dir), f"{resultname}.png"), dpi=300)

if opt.show:
    plt.show()
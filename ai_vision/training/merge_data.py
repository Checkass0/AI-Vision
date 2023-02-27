#!/usr/bin/python3
import argparse
import sys
import os
import shutil
import json

parser = argparse.ArgumentParser(description = "Merges two data directories. The second directory is empty afterwards. Duplicates get overwritten.\n"
                                                "Finds the upper most labels.txt files in both directories, then merges and sorts them.",
                                formatter_class=argparse.RawTextHelpFormatter)

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


def move_files(origin, destination):
    """moves files from origin directory to destination directory"""
    directories = sorted(os.listdir(origin))
    if not os.path.exists(destination):
        os.mkdir(destination)
    for dir in directories:
        if os.path.isdir(os.path.join(origin, dir)):
            move_files(os.path.join(origin, dir), os.path.join(destination, dir))
        else:
            try:
                shutil.move(os.path.join(origin, dir), os.path.join(destination, dir))
            except Exception as e:
                print("couldn't move file ", dir, ", an excpetion occured: ", e)

def search_labels(origin):
    """searches upper most labels.txt file and reads it into labels_list"""
    global labels_list
    directories = sorted(os.listdir(origin))
    if labels_list != None:
        return
    if "labels.txt" in directories:
            print("found labels.txt in ", origin)
            labels_file = open(os.path.join(origin, "labels.txt"), "r")
            labels = labels_file.read()
            labels_list = labels.replace("\r","\n").split("\n")
            os.remove(os.path.join(origin, "labels.txt"))
            return
    for dir in directories:
        if os.path.isdir(os.path.join(origin, dir)):
           search_labels(os.path.join(origin, dir))


while True:
    dir_1 = input("Please enter the Directory to merge data to: ")
    if not dir_1.startswith('/'):
        dir_1 = os.path.join(ai_vision_dir, "data", dir_1)

    if not os.path.isdir(os.path.join(ai_vision_dir, "data", dir_1)):
        print(f"Directory '{dir_1}' doesn't exist")
    elif len(os.listdir(dir_1)) == 0:
        print(f"Directory '{dir_1}' is empty")
    else:
        break

while True:
    dir_2 = input("Please enter the Directory to merge data from: ")
    if not dir_2.startswith('/'):
        dir_2 = os.path.join(ai_vision_dir, "data", dir_2)

    if not os.path.isdir(os.path.join(ai_vision_dir, "data", dir_2)):
        print(f"Directory '{dir_2}' doesn't exist")
    elif len(os.listdir(dir_2)) == 0:
        print(f"Directory '{dir_2}' is empty")
    else:
        break
    

# get top level folders
data_structure_1 = [item for item in os.listdir(dir_1) if os.path.isdir(os.path.join(dir_1, item))]
print(data_structure_1)
data_structure_2 = [item for item in os.listdir(dir_2) if os.path.isdir(os.path.join(dir_2, item))]
print(data_structure_2)

# check if data structures are compatible
if all(item in ["test", "train", "val"] for item in data_structure_1) and all(item in ["test", "train", "val"] for item in data_structure_2):
    print("merging two data directories with 'test-train-val' data structure")
elif (not any(item in ["test", "train", "val"] for item in data_structure_1)) and (not any(item in ["test", "train", "val"] for item in data_structure_2)):
     print("merging two data directories without 'test-train-val' data structure")
else:
    print("the data structures of the two directories are not compatible.\n"
        "please make sure the 'test-train-val' data structure exists either in both or in neither of the directories.\n"
        "exiting program...")
    sys.exit(0)

# search for labels.txt files and merge them
labels_list = None
search_labels(dir_2)
labels_2 = labels_list

labels_list = None
search_labels(dir_1)
labels_1 = labels_list

if labels_1  == None:
    newlabels = labels_2
elif labels_2 == None:
    newlabels = labels_1
else:
    newlabels = labels_1 + labels_2

if newlabels == None:
    print("no labels.txt file was found, moving files now...")
else:
    newlabels = list(dict.fromkeys(newlabels))
    newlabels = [element for element in newlabels if element != ""]     # remove empty elements
    newlabels = sorted(newlabels)
    print("merged labels.txt files to: ", newlabels)

    with open(os.path.join(dir_1, "labels.txt"), 'w') as file:
        for element in newlabels:
            file.write("%s\n" % element)

# move files recursively
move_files(dir_2, dir_1)

print("Finished merging folders, exiting program...")
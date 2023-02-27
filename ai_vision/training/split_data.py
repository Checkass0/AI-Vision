#!/usr/bin/python3
import argparse
import sys
import os
import shutil
import random
import json

parser = argparse.ArgumentParser(description = "splits data in train, val and test directories",
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

# ask for directory
while True:
    data_dir = input("Please enter the directory, you wish to split into 'test', 'train' and 'val': ")

    # check if file path for data dir was specified
    if not data_dir.startswith('/'):
        data_dir = os.path.join(ai_vision_dir, "data", data_dir)

    # check if data_dir exists
    if not os.path.isdir(data_dir):
        print(f"Directory '{data_dir}' doesn't exist")
    else:
        # remove empty directories
        directories = sorted(os.listdir(data_dir))
        for dir in directories:
            if not os.path.isdir(os.path.join(data_dir, dir)):
                directories.remove(dir)

        if len(directories) == 0:
            print(f"Directory  '{data_dir}' is empty")
        # check if directories test, val, train exist
        elif any(item in ["test", "train", "val"] for item in directories):
                print("please make sure directories 'test','train','val' don't exist in the data directory.\n"
                    "they will be created in the process\n")
        else:
            break

while True:
    # ask for test percentage
    while True:
        test_percentage = input("Percentage of test images (For Default press ENTER): ")
        if test_percentage == "":
            test_percentage = 10
            break
        elif test_percentage.isnumeric():
            test_percentage = int(test_percentage)
            if test_percentage >= 0 and test_percentage <= 100: 
                break
            else:
                print("Please enter a Number between 0 and 100")
        else:
            print("Please enter a Number between 0 and 100")

    # ask for val percentage
    while True:
        val_percentage = input("Percentage of validation images (For Default press ENTER): ")
        if val_percentage == "":
            val_percentage = 10
            break
        elif val_percentage.isnumeric():
            val_percentage = int(val_percentage)
            if val_percentage >= 0 and val_percentage <= 100: 
                break
            else:
                print("Please enter a Number between 0 and 100")
        else:
            print("Please enter a Number between 0 and 100")

    # check sum of test and val
    if test_percentage + val_percentage > 100:
        print("Sum of test and validation can't be more than 100")
    else:
        break

try:
    input(f"splitting output into: {test_percentage}% test, {100-test_percentage-val_percentage}% train, {val_percentage}% val\n"
        "Press any key to continue, press 'Ctrl+C' to abort")
except KeyboardInterrupt:
    print("")
    sys.exit(0)

# create test, val, train directories
os.mkdir(os.path.join(data_dir, "test"))
os.mkdir(os.path.join(data_dir, "train"))
os.mkdir(os.path.join(data_dir, "val"))

test_dir = os.path.join(data_dir, "test")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# create empty directories in test and val directories
for dir in directories:
    os.mkdir(os.path.join(test_dir, dir))
    os.mkdir(os.path.join(val_dir, dir))

# move random files to train and val directories
for dir in directories:
    print("directory opened: ", dir)
    image_list = os.listdir(os.path.join(data_dir, dir))
    number_of_images = len(image_list)
    print("total number of images: ", number_of_images)

    # create random image lists
    number_of_test_images = int(number_of_images*test_percentage/100)
    number_of_val_images = int(number_of_images*val_percentage/100)
    print("number of test images: ", number_of_test_images)
    print("number of val images: ", number_of_val_images)

    random_images = random.sample(image_list, number_of_test_images + number_of_val_images)
    random_test_images = random_images[0:number_of_test_images]
    random_val_images = random_images[number_of_test_images:]

    # move images to test directory
    for rand_img in random_test_images:
        if os.path.exists(os.path.join(data_dir, dir, rand_img)):
            shutil.move(os.path.join(data_dir, dir, rand_img), os.path.join(test_dir, dir, rand_img))
        else:
            print("file ", os.path.join(data_dir, dir, rand_img), "doesn't exist")
    
    # move images to val directory
    for rand_img in random_val_images:
        if os.path.exists(os.path.join(data_dir, dir, rand_img)):
            shutil.move(os.path.join(data_dir, dir, rand_img), os.path.join(val_dir, dir, rand_img))
        else:
            print("file ", os.path.join(data_dir, dir, rand_img), "doesn't exist")

# move remaining data to train directory
for dir in directories:
    shutil.move(os.path.join(data_dir, dir), os.path.join(data_dir, "train", dir))

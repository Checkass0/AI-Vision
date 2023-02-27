#!/usr/bin/python3
import argparse
import sys
import os
import json

# parse the command line
parser = argparse.ArgumentParser(description = "helper to create a new configuration",
                                formatter_class=argparse.RawTextHelpFormatter)

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


dirname = os.path.dirname(os.path.realpath(__file__))
dirname = os.path.dirname(dirname)


# get neural network config
with open(os.path.join(dirname, "ressource", "neural_network_config.json")) as f:
    neural_network_config = json.load(f)
ai_vision_dir = os.path.expanduser(neural_network_config["ai_vision_dir"])

settings_dir = os.path.join(ai_vision_dir, "settings")

# get connector name from user
housing = input("Please enter the name of the housing: ")

# create housing directory if not existent
if not os.path.exists(os.path.join(settings_dir, housing)):
    print(f"The housing '{housing}' doesn't exist, creating it now...")
    os.mkdir(os.path.join(settings_dir, housing))

# get configuration name from user
configuration = input("Please enter the name of the configuration: ")
configuration = os.path.splitext(configuration)[0]

if os.path.exists(os.path.join(settings_dir, housing, configuration + '.json')):
    print(f"Configuration {configuration} already exists, please edit it manually with a text editor")
    sys.exit(0)

data = {
    "name": configuration,
    "plug": housing,
    "holes": []
}


print("Press 'Ctrl+C' to finish")

# append user input to data
while True:
    try:
        print("--------------")
        color = input("Color: ")

        while True:
            number = input("Number: ")
            try:
                number = int(number)
                break
            except Exception as e :
                print('\033[A                            \033[A')
        # write to json
        data["holes"].append({
            "expected": color,
            "chamber": number
        })
    except KeyboardInterrupt:
        print(f"\nsaving configuration {configuration}...")
        with open(os.path.join(settings_dir, housing, configuration + '.json'), 'w') as f:
            json.dump(data, f, indent=4)
        sys.exit(0)


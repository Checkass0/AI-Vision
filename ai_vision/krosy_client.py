#!/usr/bin/python3

# send json string to localhost on port 5000
# adjust this for standalone mode later (read json file and send as string)

import requests
import os
import json

dirname = os.path.dirname(os.path.realpath(__file__))

class KrosyClient():
    """simulates request by krosy"""
    def __init__(self):
        # self.url = "http://localhost:5000/data"

        # load settings from network config file
        with open(os.path.join(dirname, "ressource", "network_config.json")) as f:
            network_config = json.load(f)
        self.url = network_config["url"]        

    def send_data(self, configurationfile):
        """sends data from file"""
        # load expected colors and sequence
        with open(configurationfile, "r") as f:
            json_data = json.load(f)
        print(json_data)

        res = requests.post(self.url, json=json_data)

if __name__ == '__main__':
    client = KrosyClient()
    client.send_data("/home/ai-vision/AI_Vision/settings/P24096HK/N2_5_1-B_V1.json")
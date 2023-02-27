#!/usr/bin/python3

from flask import Flask, request
import requests
import threading
import os
import json

class KrosyServer(threading.Thread):
    """creates a webserver, writes received data to json file"""
    app = Flask(__name__)
    krosy_dir = None
    nn_thread = None


    def __init__(self, nn_thread, krosy_dir):
        super(KrosyServer, self).__init__()
        KrosyServer.nn_thread = nn_thread  # reference to the neural network thread, used for setting events
        KrosyServer.krosy_dir = krosy_dir

    def run(self):
        self.app.run(debug=False, host='0.0.0.0', port='5000')

    def stop(self):
        print("Shutting down Server...")
        resp = requests.get("http://localhost:5000/shutdown")

    @app.route("/shutdown")
    def shutdown():
        shutdown_func = request.environ.get('werkzeug.server.shutdown')
        if shutdown_func is None:
            raise RuntimeError("Not running werkzeug")
        shutdown_func()
        return "Shutting down..."
        
    @app.route('/')
    def hello_world():
        print("Device opened Website")
        print(KrosyServer.krosy_dir)
        return "Hello World"

    @app.route('/data/', methods=['POST'])
    def get_json_data():
        data = request.json
        KrosyServer.nn_thread.closeDetection()

        # write json data to file
        with open(os.path.join(KrosyServer.krosy_dir, "data.json"), "w+") as f:
            json.dump(data, f)

        KrosyServer.nn_thread.start_detection_event.set()
        return "sucess"

if __name__ == "__main__":
    server = KrosyServer(None, "/home/nvidia/AI_Vision/krosy")
    server.start()
    


import jetson.utils
import os
import json

dirname = os.path.dirname(os.path.realpath(__file__))

# load settings from network config file
with open(os.path.join(dirname, "ressource", "neural_network_config.json")) as f:
    network_config = json.load(f)
video_source = network_config["video_source"]
video_sink = network_config["video_sink"]
network = network_config["network"]
model = network_config["model"]
labels = network_config["labels"]
input_blob = network_config["input_blob"]
output_blob = network_config["output_blob"]
width = network_config["width"]
height = network_config["height"]

# create video sources & outputs
input = jetson.utils.videoSource(video_source, argv=[f"--input-width={width}", f"--input-height={height}", "--input-flip=rotate-180"])
output = jetson.utils.videoOutput("display://0", argv=[])

img = input.Capture()
while output.IsStreaming() and input.IsStreaming():
    img = input.Capture()
    output.Render(img)

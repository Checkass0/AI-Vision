import jetson.utils
import json
import numpy as np

# def load_configuration(configurationfile):
#     """Loads desired configuration for connector"""
#     expected_colors = []
#     with open(configurationfile, "r") as f:
#         json_data = json.load(f)

#     for idx in json_data["holes"]:
#         expected_colors.append(json_data["holes"][str(idx)]["expected"])
#     return expected_colors

def load_connector(connectorfile):
    """Loads connector from csv file and returns dimensions for cropping as array of [left,top,right,bottom]"""
    try:
        connector_array = np.genfromtxt(connectorfile, delimiter = ',', comments='#')
        left = connector_array[:,0] - connector_array[:,2]/2
        top = connector_array[:,1] - connector_array[:,3]/2
        right = connector_array[:,0] + connector_array[:,2]/2
        bottom = connector_array[:,1] + connector_array[:,3]/2
        return[left,top,right,bottom]
    except Exception as e: print(e)

def crop_images(left, top, right, bottom, img_original):
    """crops out images according to dimensions array"""
    count = 0
    try:
        images = []
        for l,t,r,b in zip(left,top,right,bottom):
            count +=1
            crop_roi = (l,t,r,b)
            img_cropped = jetson.utils.cudaAllocMapped(width=r-l, height = b-t, format = img_original.format)
            jetson.utils.cudaCrop(img_original, img_cropped, crop_roi)
            images.append(img_cropped)
        return images
    except Exception as e: print(e)

def create_overlay(left, top, right, bottom, img):
    """creates rectange overlay for adjusting connector chambers"""
    for l,t,r,b in zip(left, top, right, bottom):
        jetson.utils.cudaDrawLine(img, (l,t), (l,b), (0,0,0,255), 1)
        jetson.utils.cudaDrawLine(img, (l,b), (r,b), (0,0,0,255), 1)
        jetson.utils.cudaDrawLine(img, (r,b), (r,t), (0,0,0,255), 1)
        jetson.utils.cudaDrawLine(img, (r,t), (l,t), (0,0,0,255), 1)

def create_expected_result(json_file):
    """returns lists of expected results and chambers ordered by sequence"""
    expected_results = [holes["expected"] for holes in json_file["holes"]]
    chambers = [holes["chamber"] for holes in json_file["holes"]]
    return expected_results, chambers
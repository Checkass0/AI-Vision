import jetson.utils
import numpy as np
import json
from PIL import Image, ImageOps, ImageDraw
from PIL.ImageQt import ImageQt
import os
from logWidget import LogWidget

logging = LogWidget()

def load_connector(connectorfile: str):
    """Loads connector from csv file and returns dimensions for cropping as array of [left,top,right,bottom]"""
    try:
        connector_array = np.genfromtxt(connectorfile, delimiter = ',', comments='#')
        left = connector_array[:,0] - connector_array[:,2]/2
        top = connector_array[:,1] - connector_array[:,3]/2
        right = connector_array[:,0] + connector_array[:,2]/2
        bottom = connector_array[:,1] + connector_array[:,3]/2
        return[left,top,right,bottom]
    except Exception as e: print(e)

def crop_image(left: int, top: int, right: int, bottom: int, img_original):
    """crops out images according to dimensions array"""
    try:
        crop_roi = (left,top,right,bottom)
        img_cropped = jetson.utils.cudaAllocMapped(width=right-left, height = bottom-top, format = img_original.format)
        jetson.utils.cudaCrop(img_original, img_cropped, crop_roi)
        return img_cropped
    except Exception as e: print(e)

def create_overlays(positions: list, colors: tuple, img):
    """creates rectange overlays for connector chambers"""
    for position, color in zip(positions, colors):
        left = position[0]
        top = position[1]
        right = position[2]
        bottom = position[3]
        jetson.utils.cudaDrawLine(img, (left,top), (left,bottom), color, 1)
        jetson.utils.cudaDrawLine(img, (left,bottom), (right,bottom), color, 1)
        jetson.utils.cudaDrawLine(img, (right,bottom), (right,top), color, 1)
        jetson.utils.cudaDrawLine(img, (right,top), (left,top), color, 1)

def create_expected_result(json_file):
    """returns lists of expected results and chambers ordered by sequence"""
    expected_results = [holes["expected"] for holes in json_file["holes"]]
    chambers = [holes["chamber"] for holes in json_file["holes"]]
    return expected_results, chambers

def createWireColorImages(path: str, color_str: str, height: int):
    "returns active and inactive images for wire color"
    with open(path) as f:
        json_colors = json.load(f)
    default_color = json_colors['default']

    color_list = color_str.split("_")

    try:
        color_code = []
        for color in color_list:
            if os.path.exists(os.path.join(os.path.dirname(path), "classes", color_str + ".jpg")):
                print(f"loading color {color}")
                with Image.open(os.path.join(os.path.dirname(path), "classes", color_str + ".jpg")) as img:
                    img = img.resize((int(height*9/10),int(height*9/10)))
                color_list = []
                break
            elif color in json_colors:
                color_code.append(json_colors[color])
            else:
                print(f"not loading color {color}")
                raise Exception(f"color '{color}' doesn't exist in gui_config.json nor in 'classes' directory")

        if len(color_list) == 0:
            pass
        elif len(color_list) == 1:
            img = Image.new('RGB', (int(height*9/10),int(height*9/10)), color_code[0])
        elif len(color_list) == 2:
            img1 = Image.new('RGB', (int(height*9/10),int(height*4.5/10)), color_code[0])
            img = Image.new('RGB', (int(height*9/10),int(height*9/10)), color_code[1])
            img.paste(img1)
        elif len(color_list) == 3:
            img1 = Image.new('RGB', (int(height*9/10),int(height*3/10)), color_code[0])
            img2 = Image.new('RGB', (int(height*9/10),int(height*6/10)), color_code[1])
            img = Image.new('RGB', (int(height*9/10),int(height*9/10)), color_code[2])
            img.paste(img2)
            img.paste(img1)
        else:
            raise Exception("too many colors given, the maximum is 3")
    except Exception as e:
        logging.set_error(1, str(e))
        # print(f"Excpetion '{str(e)}' occured during creation of the wire color image, filling with error picture")
        with Image.open(os.path.join(os.path.dirname(path), "classes", "not-available.jpg")) as img:
            img = img.resize((int(height*9/10),int(height*9/10)))

    
    active_img = ImageOps.expand(img, border=int(height/20), fill='white')
    inactive_img = ImageOps.expand(img, border=int(height/20), fill=default_color)
     
    return ImageQt(active_img), ImageQt(inactive_img)

def get_rotation(rotate_180: bool, rotate_90: bool):
    "returns the string for rotation"
    if not rotate_180 and not rotate_90:
        return "none"
    elif not rotate_180 and rotate_90:
        return "clockwise"
    elif rotate_180 and not rotate_90:
        return "rotate-180"
    else:
        return "counterclockwise" 
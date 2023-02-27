import collections
import os

from helpers import createWireColorImages
from logWidget import LogWidget
import jetson.utils

class Chamber():
    """class of chamber objects"""
    chunk_size = 1
    height = 300
    dirname = None
    remaining_chambers = []
    parallel_chambers = []

    def __init__(self, sequence, chamberNumber, position, expected_result, all_available_results, threshold = 0.6, length = 10):
        self.logging = LogWidget()
        if expected_result not in all_available_results:
            self.logging.set_error(1, f"The requested color {expected_result} was not trained. It cannot be detected")
            expected_result = "error"
        # create parallel_chambers and remaining_chambers list
        if len(Chamber.parallel_chambers) < Chamber.chunk_size:
            Chamber.parallel_chambers.append(self)
        else:
            Chamber.remaining_chambers.append(self)
        
        # constants
        self.sequence = sequence
        self.chamberNumber = chamberNumber
        self.position = position
        self.expected_result = expected_result
        self.threshold = threshold
        self.length = length
        self.active_image, self.inactive_image = createWireColorImages(os.path.join(Chamber.dirname, "ressource", "color_config.json"), expected_result, Chamber.height)

        # variables
        self.active = False
        self.result = None
        self.confidence = 0
        self.last_confidence = 0    # debugging
        self.last_result = "None"   # debugging
        self.overlay_color = (0,0,0,150)
        self.resultBuffer = collections.deque(["occluded"]*self.length, maxlen = self.length)
        self.confidenceBuffer = collections.deque([1]*self.length, maxlen = self.length)
        self.active_lock = False

    def get_result(self):
        return self.result

    def get_overlay_color(self):
        return self.overlay_color

    def get_sequence(self):
        return self.sequence

    def get_chamber_number(self):
        return self.chamberNumber

    def get_position(self):
        return self.position

    def get_correct(self):
        return self.result == self.expected_result

    def get_expected_result(self):
        return self.expected_result

    def get_active(self):
        return self.active
    
    def get_images(self):
        return self.active_image, self.inactive_image

    def set_active(self, active):
        if not self.active_lock:
            self.active = active
        self.set_overlay_color()

    def set_correct(self):
        self.result = self.expected_result
        try:
            Chamber.parallel_chambers.remove(self)
            Chamber.parallel_chambers.append(Chamber.remaining_chambers.pop(0))
        except: pass
        self.set_overlay_color()

    def set_active_lock(self):
        """deactivates setting overlay color to active"""
        self.active_lock = True
        self.active = False
        self.set_overlay_color()

    def set_result(self, result, confidence):
        """evaluates detection and sets result"""
        self.last_confidence = confidence   # debugging
        self.last_result = result           # debugging
        if confidence > self.threshold:
            self.resultBuffer.append(result)
            self.confidenceBuffer.append(confidence)
        self.result = max(self.resultBuffer, key = self.resultBuffer.count)
        
        if self.get_correct():
            self.set_active_lock()
            self.set_correct()
            confidenceList = [confidence for result, confidence in zip(self.resultBuffer, self.confidenceBuffer) if result == self.expected_result]
            self.confidence = sum(confidenceList) / len(confidenceList)
            self.logging.set_info(1, f"Chamber {self.expected_result} is correct, confidence: {self.confidence*100}%")

        self.set_overlay_color()

    def set_overlay_color(self):
        """sets overlay_color according to result and expected_result"""
        if self.active:
            self.overlay_color = (255,255,255,150)
        elif self.result == self.expected_result:
            self.overlay_color = (92,214,92,150)
        elif self.result == "occluded" or self.result == "empty":
            self.overlay_color = (0,0,0,150)
        else:
            self.overlay_color = (255,51,51,150)

    def set_overlay(self, img):
        """renders the overlay box on the img"""
        
        # left = self.position[0]
        # top = self.position[1]
        # right = self.position[2]
        # bottom = self.position[3]
        jetson.utils.cudaDeviceSynchronize()
        # jetson.utils.cudaDrawLine(img, (left,top), (left,bottom), self.overlay_color, 1)
        # jetson.utils.cudaDrawLine(img, (left,bottom), (right,bottom), self.overlay_color, 1)
        # jetson.utils.cudaDrawLine(img, (right,bottom), (right,top), self.overlay_color, 1)
        # jetson.utils.cudaDrawLine(img, (right,top), (left,top), self.overlay_color, 1)
        jetson.utils.cudaDrawRect(img, tuple(self.position), self.overlay_color)

from helpers import *
from PyQt5.QtGui import QPixmap


class ErrorPicture():
    """class that defines an Error Button"""
    active_picture = -1   

    def __init__(self, position: int, height: float, label: QPixmap, config: str):
        # constants
        self.position = position
        _default_image_active, self.default_image = createWireColorImages(config, "default", height)

        # variables
        self.last_confidence = 0    # debugging
        self.last_result = "None"   # debugging
        self.label = label
        self.active_image = self.default_image
        self.inactive_image = self.default_image

        self.clicked = False
        self.clickable = False
        self.active = False

        # set default image
        image = QPixmap.fromImage(self.default_image)
        self.label.setPixmap(image)

    def get_image(self):
        if self.active:
            image = QPixmap.fromImage(self.active_image)
        else:
            image = QPixmap.fromImage(self.inactive_image)
        self.label.setPixmap(image)
    
    def get_clickable(self):
        return self.clickable

    def get_clicked(self):
        return self.clicked

    def get_active(self,):
        return self.active
    
    def set_error(self):
        pass

    def set_images(self, active_image, inactive_image):
        self.clickable = True
        self.active_image = active_image
        self.inactive_image = inactive_image

    def set_clicked(self):
        if self.clickable:
            self.clicked = True
            ErrorPicture.active_picture = self.position

    def set_unclicked(self):
        self.clicked = False
        self.active = False

    def set_active(self):
        if self.clicked and self.clickable:
            self.active = True

    def set_inactive(self):
        self.active = False
            
    def set_reset(self):
        self.clickable = False
        self.clicked = False
        self.active = False
        self.active_image = self.default_image
        self.inactive_image = self.default_image
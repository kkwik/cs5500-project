import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import copy
import hashlib
from PIL import Image
import numpy as np
import util


class Watermark(ImageOperationInterface):
    watermark = np.zeros(1)

    @staticmethod
    def name():
        return 'WATERMARK'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Text("Select Watermark Image", key="watermark_image"),
                sg.In(size=(30,1), enable_events=True, key="WATERMARK_SELECTED"),
                sg.FileBrowse()
            ],
            [
                sg.Button('Insert', key=f'{Watermark.name()}_INSERT', disabled=True)
            ]
        ]

        gui = sg.Column(contents)

        return gui

    def getMd5(input: str) -> str:
        return hashlib.md5(str.encode('utf-8')).hexdigest()

    # Operations
    @staticmethod
    def insert(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        return copy.deepcopy(original)
    
    @staticmethod
    def watermark_selected(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        window[f'{Watermark.name()}_INSERT'].update(disabled=False) # Enable inserting watermark

        # Load watermark into binary image and store in static variable
        filename = values["WATERMARK_SELECTED"]
        
        input_image = Image.open(filename)      # Load image
        input_image = input_image.convert('1') # Convert to binary mage
        Watermark.watermark = np.asarray(input_image) # Extract pixel values as array


        return copy.deepcopy(modified)

    # Events
    @staticmethod
    def get_operation(operation_name: str) -> callable:
        if operation_name == f'{Watermark.name()}_INSERT':
            return Watermark.insert
        elif operation_name == f'{Watermark.name()}_SELECTED':
            return Watermark.watermark_selected
        else:
            print("Unknown watermark operation")
        return

def get_instance():
    return Watermark()
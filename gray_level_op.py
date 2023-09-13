import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np


class GrayResolution(ImageOperationInterface):
    @staticmethod
    def name():
        return 'GRAY'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Slider(range=(1,8), default_value=8, size=(20,15), orientation='horizontal', key='GRAY_SLIDER')
            ],
            [
                sg.Button("Apply", key="GRAY_APPLY"), 
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Operations
    @staticmethod
    def gray_apply(original, modified, window, values):
        print("GRAY APPLY")

        source_gray_res = modified['gray_resolution']
        desired_gray_res = int(values['GRAY_SLIDER'])

        # Bit conversion
        convert = lambda val: int(f'{val:08b}'[:desired_gray_res], 2) # Take val, turn it into 8-bit binary string. Get the amount of lsb we want, turn back into an int
        convert_vect = np.vectorize(convert)                         
        modified['image'] = convert_vect(modified['image']).astype(np.uint8) 

        # Distributing values across range 0-255 so that we can actually see the image at low bit resolution
        modified['image'] = (255 / np.max(modified['image'])) * modified['image']   # Scale bits back to full range so we'd have 0 and 255 instead of 0 and 1
        modified['image'] = np.floor(modified['image']).astype(np.uint8)            # Normalize values into valid pixel values
        modified['gray_resolution'] = desired_gray_res

        print(np.max(modified['image']))

        
        return modified

    # Events
    @staticmethod
    def get_operation(operation_name):
        
        if operation_name == f'{GrayResolution.name()}_APPLY':
            return GrayResolution.gray_apply
        else:
            print("Unknown gray operation")
        return

def get_instance():
    return GrayResolution()

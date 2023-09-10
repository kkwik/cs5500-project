import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np

class Zoom(ImageOperationInterface):
    @staticmethod
    def name():
        return 'ZOOM'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Combo(['Nearest Neighbor', 'Linear', 'Bilinear'], default_value='Nearest Neighbor', key='ZOOM_TYPE')
            ],
            [
                sg.Slider(range=(1,200), default_value=100, size=(20,15), orientation='horizontal', key='ZOOM_SLIDER')
            ],
            [
                sg.Button("Apply", key="ZOOM_APPLY"), 
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Operations
    @staticmethod
    def zoom_apply(original, modified, values):
        print("ZOOM APPLY")
        print(values)
        return np.transpose(modified)

    # Events
    @staticmethod
    def get_operation(operation_name):
        
        if operation_name == f'{Zoom.name()}_APPLY':
            return Zoom.zoom_apply
        else:
            print("Unknown zoom operation")
        return

def get_instance():
    return Zoom()

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
                sg.Text("Zoom Operation"),
            ],
            [
                sg.Button("+", key="ZOOM_IN"), 
                sg.Button("-", key="ZOOM_OUT")
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Operations
    @staticmethod
    def zoom_in(original, modified):
        print("ZOOM IN")
        return np.transpose(modified)

    @staticmethod
    def zoom_out(original, modified):
        print("ZOOM OUT")
        return np.transpose(modified)

    # Events
    @staticmethod
    def get_operation(operation_name):
        
        if operation_name == f'{Zoom.name()}_IN':
            return Zoom.zoom_in
        elif operation_name == f'{Zoom.name()}_OUT':
            return Zoom.zoom_out
        else:
            print("Unknown zoom operation")
        return

def get_instance():
    return Zoom()

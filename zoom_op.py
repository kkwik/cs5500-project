import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface


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
    def zoom_in(input):
        print("ZOOM IN")
        return ""

    def zoom_out(input):
        print("ZOOM OUT")
        return ""

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

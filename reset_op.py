import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface


class Reset(ImageOperationInterface):
    @staticmethod
    def name():
        return 'RESET'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Button('Reset ↺', key="RESET")
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Operations
    @staticmethod
    def reset(original, modified, window, values):
        return original

    # Events
    @staticmethod
    def get_operation(operation_name):
        if operation_name == f'{Reset.name()}':
            return Reset.reset
        else:
            print("Unknown reset operation")
        return

def get_instance():
    return Reset()
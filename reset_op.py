import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import copy


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
    def reset(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        return copy.deepcopy(original)

    # Events
    @staticmethod
    def get_operation(operation_name: str) -> callable:
        if operation_name == f'{Reset.name()}':
            return Reset.reset
        else:
            print("Unknown reset operation")
        return

def get_instance():
    return Reset()
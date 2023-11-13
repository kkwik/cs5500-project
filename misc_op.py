import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import copy
from PIL import Image


class Misc(ImageOperationInterface):
    @staticmethod
    def name():
        return 'MISC'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Button('Reset ↺', key=f'{Misc.name()}_RESET'),
                sg.Button('Subtract', key=f'{Misc.name()}_SUBTRACT')
            ],
            [
                sg.Text("File Name:"),
                sg.Input('tmp.png', key=f'{Misc.name()}_FILE_NAME'),
                sg.Button('Save', key=f'{Misc.name()}_SAVE')
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Operations
    @staticmethod
    def reset(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        return copy.deepcopy(original)

    @staticmethod
    def subtract(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        modified['image'] -= original['image']
        return modified

    @staticmethod
    def save(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        image = Image.fromarray(modified['image'])
        image.save(values[f'{Misc.name()}_FILE_NAME'])
        return modified

    # Events
    @staticmethod
    def get_operation(operation_name: str) -> callable:
        if operation_name == f'{Misc.name()}_RESET':
            return Misc.reset
        elif operation_name == f'{Misc.name()}_SUBTRACT':
            return Misc.subtract
        elif operation_name == f'{Misc.name()}_SAVE':
            return Misc.save
        else:
            print("Unknown misc operation")
        return

def get_instance():
    return Misc()
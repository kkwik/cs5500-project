import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np


class BitPlane(ImageOperationInterface):
    @staticmethod
    def name():
        return 'Bit_Plane'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Text('Bit Plane Selector')
            ],
            [
                sg.Slider(range=(0, 7), default_value=0, size=(20,15), orientation='horizontal', key=f'{BitPlane.name()}_PLANE')
            ],
            [
                sg.Button('Show', key=f'{BitPlane.name()}_SHOW_APPLY'),
            ],
            [
                sg.Button('Hide', key=f'{BitPlane.name()}_HIDE_APPLY'),
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Operations
    @staticmethod
    def show_bit_plane(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        plane_to_show = int(values[f'{BitPlane.name()}_PLANE'])

        source_img = modified['image'] 
        Y, X = source_img.shape # Y, X
        result_img = np.empty(source_img.shape)
        selected_plane = 2**plane_to_show

        for y in range(Y):
            for x in range(X):
                result_img[y,x] = 0 if (source_img[y,x] & selected_plane) == 0 else 255

        modified['image'] = result_img.astype(np.uint8) 
        return modified

    @staticmethod
    def hide_bit_plane(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        plane_to_hide = int(values[f'{BitPlane.name()}_PLANE'])

        source_img = modified['image'] 
        Y, X = source_img.shape # Y, X
        result_img = np.empty(source_img.shape)
        selected_plane = 2**plane_to_hide

        for y in range(Y):
            for x in range(X):
                result_img[y,x] = source_img[y,x] ^ selected_plane if (source_img[y,x] & selected_plane) == selected_plane else source_img[y,x]

        modified['image'] = result_img.astype(np.uint8) 
        return modified

    # Events
    @staticmethod
    def get_operation(operation_name: str) -> callable:
        if operation_name == f'{BitPlane.name()}_SHOW_APPLY':
            return BitPlane.show_bit_plane
        elif operation_name == f'{BitPlane.name()}_HIDE_APPLY':
            return BitPlane.hide_bit_plane
        else:
            print("Unknown BitPlane operation")
        return

def get_instance():
    return BitPlane()
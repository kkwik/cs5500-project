import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np
from math import floor
np.set_printoptions(threshold=np.inf)

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
    def find_nearest(source, pixel, factor):
        source_mapped_pixel = np.floor(pixel / factor).astype(int)  # Map the pixel location in the upscaled image to the closest pixel in the source image
        return source[source_mapped_pixel[0]][source_mapped_pixel[1]]   # Retrieve the mapped pixel value

    @staticmethod
    def zoom_nn(start_image, factor):
        old_dim = np.asarray(start_image.shape)
        new_dim = (factor * old_dim).round().astype(np.int64)

        res = np.zeros(new_dim)
        res[:] = -1 # Give a nonsense value so we know which values still need to be filled in
        
        # i == y, j == x
        # Transfer real values to new size
        for i in range(old_dim[0]):
            for j in range(old_dim[1]):
                res[round(factor * i)][round(factor * j)] = start_image[i][j]

        

        # Fill in gaps
        for i in range(new_dim[0]):
            for j in range(new_dim[1]):
                if res[i][j] == -1:
                    res[i][j] = Zoom.find_nearest(start_image, np.array([i,j]), factor)

        return res.astype(np.uint8) # Convert to uint8 at the end. Note: if there are any -1's leftover they will be wrapped to 255

    @staticmethod
    def zoom_linear(start_image, factor):
        return

    @staticmethod
    def zoom_bilinear(start_image, factor):
        return
    


    @staticmethod
    def zoom_apply(source_image, working_image, values):
        print("ZOOM APPLY")
        print(values)


        factor = values['ZOOM_SLIDER'] / 100
        result_image = working_image['image']

        if values['ZOOM_TYPE'] == 'Nearest Neighbor':
            result_image = Zoom.zoom_nn(working_image['image'], factor)
        elif values['ZOOM_TYPE'] == 'Linear':
            result_image = Zoom.zoom_linear(working_image['image'], factor)
        elif values['ZOOM_TYPE'] == 'Bilinear':
            result_image = Zoom.zoom_bilinear(working_image['image'], factor)
        else:
            print('ERROR: unknown type of zoom requested')

        working_image['image'] = result_image
        
        return working_image

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

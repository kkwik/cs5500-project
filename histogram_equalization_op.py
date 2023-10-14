import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np
import copy
import itertools
import math
import numpy.typing as npt

class HistEqual(ImageOperationInterface):
    @staticmethod
    def name():
        return 'HISTOGRAM_EQUALIZATION'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Text('Global Histogram Equalization')
            ],
            [
                sg.Button('Apply', key=f'{HistEqual.name()}_GLOBAL')
            ],
            [
                sg.HorizontalSeparator()
            ],
            [
                sg.Text('Local Histogram Equalization')
            ],
            [
                sg.Text('Mask Size: ')
            ],
            [
                sg.Slider(range=(1, 51), default_value=9, size=(20,15), orientation='horizontal', resolution=2, key=f'{HistEqual.name()}_LOCAL_SIZE')
            ],
            [
                sg.Button('Apply', key=f'{HistEqual.name()}_LOCAL'),
            ]
        ]

        gui = sg.Column(contents)

        return gui

    @staticmethod
    def hist_equal(img_data: npt.NDArray, bit_depth: int) -> npt.NDArray:
        
        # Count frequencies
        frequencies = {}
        index, count = np.unique(img_data.flatten(), return_counts=True)
        for rk, f in zip(index, count):
            frequencies[rk] = f
        
        # Calculate PDF
        total_size = img_data.size
        pdf = {}
        for rk in frequencies.keys():
            pdf[rk] = frequencies[rk] / total_size

        # Calculate CDF
        cdf = {}
        for i in range(len(pdf)):
            cdf[i] = round((2**bit_depth - 1) * sum(dict(itertools.islice(pdf.items(), i+1)).values()))

        # Generate HE Map
        he_map = {}
        for rk, cf in zip(frequencies.keys(), cdf.values()):
            he_map[rk] = cf

        # Modify Values According to HE Map
        img_data = np.vectorize(he_map.get)(img_data).astype(np.uint8) 

        # print("freq: ", frequencies)
        # print("pdf: ", pdf)
        # print("cdf: ", cdf)
        # print("he_map: ", he_map)
        # print("data: ", img_data)
        return img_data


    # Operations
    @staticmethod
    def global_hist_equal(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        modified['image'] = HistEqual.hist_equal(modified['image'], modified['gray_resolution'])
        return modified

    @staticmethod
    def local_hist_equal(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        mask_size = values[f'{HistEqual.name()}_LOCAL_SIZE']

        source_img = modified['image'] # np.array([[0,1,2,3,4],[5,6,7,8,9],[0,1,2,3,4],[5,6,7,8,9],[0,1,2,3,4]])#
        Y, X = source_img.shape # Y, X

        half_mask = math.floor(mask_size / 2)

        equalized_img = np.empty(source_img.shape)

        print('total size: ', X, ', ', Y)

        for y in range(Y):
            for x in range(X):
                # Find bounds of the local neighborhood
                x_start = max(x - half_mask, 0)
                x_end =   min(x + half_mask + 1, X)
                y_start = max(y - half_mask, 0)
                y_end =   min(y + half_mask + 1, Y)

                # Find the location of the pixel we're checking for in the smaller chunk
                local_x_coord = x - x_start
                local_y_coord = y - y_start

                chunk = source_img[y_start:y_end, x_start:x_end]

                equalized_img[y,x] = HistEqual.hist_equal(chunk, modified['gray_resolution'])[local_y_coord,local_x_coord]

        modified['image'] = equalized_img.astype(np.uint8) 

        return modified

    # Events
    @staticmethod
    def get_operation(operation_name: str) -> callable:
        if operation_name == f'{HistEqual.name()}_GLOBAL':
            return HistEqual.global_hist_equal
        elif operation_name == f'{HistEqual.name()}_LOCAL':
            return HistEqual.local_hist_equal
        else:
            print("Unknown histogram equalization operation")
        return

def get_instance():
    return HistEqual()
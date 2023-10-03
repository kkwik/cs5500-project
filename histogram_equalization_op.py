import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np
import copy
import itertools


class HistEqual(ImageOperationInterface):
    @staticmethod
    def name():
        return 'Histogram Equalization'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Button('Global Histogram Equalization', key=f'{HistEqual.name()}_GLOBAL')
            ]
        ]

        gui = sg.Column(contents)

        return gui

    @staticmethod
    def hist_equal(img_data, bit_depth):
        
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

        # print("pdf: ", pdf)
        # print("cdf: ", cdf)
        # print("he_map: ", he_map)
        # print("data: ", img_data)
        return img_data


    # Operations
    @staticmethod
    def global_hist_equal(original, modified, window, values):
        modified['image'] = HistEqual.hist_equal(modified['image'], modified['gray_resolution'])
        return modified

    # Events
    @staticmethod
    def get_operation(operation_name):
        if operation_name == f'{HistEqual.name()}_GLOBAL':
            return HistEqual.global_hist_equal
        else:
            print("Unknown histogram equalization operation")
        return

def get_instance():
    return HistEqual()
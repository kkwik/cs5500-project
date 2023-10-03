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
    def hist_equal(data):
        
        # Count frequencies
        frequencies = {}
        index, count = np.unique(data['image'].flatten(), return_counts=True)
        for rk, f in zip(index, count):
            frequencies[rk] = f
        
        # Calculate PDF
        total_size = data['image'].size
        pdf = {}
        for rk in frequencies.keys():
            pdf[rk] = frequencies[rk] / total_size

        # Calculate CDF
        cdf = {}
        for i in range(len(pdf)):
            cdf[i] = round((2**data['gray_resolution'] - 1) * sum(dict(itertools.islice(pdf.items(), i+1)).values()))

        # Generate HE Map
        he_map = {}
        for rk, cf in zip(frequencies.keys(), cdf.values()):
            he_map[rk] = cf

        # Modify Values According to HE Map
        data['image'] = np.vectorize(he_map.get)(data['image']).astype(np.uint8) 

        print("pdf: ", pdf)
        print("cdf: ", cdf)
        print("he_map: ", he_map)
        print("data: ", data['image'])
        return data


    # Operations
    @staticmethod
    def global_hist_equal(original, modified, window, values):
        temp = HistEqual.hist_equal(modified)
        return temp

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
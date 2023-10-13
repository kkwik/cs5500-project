import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import math
import numpy as np
import copy
import numpy.typing as npt

class SpatialFilter(ImageOperationInterface):
    @staticmethod
    def name():
        return 'Spatial_Filter'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Combo(['Smoothing', 'Median', 'Laplacian', 'High-Boost'], default_value='Smoothing', key=f'{SpatialFilter.name()}_TYPE', enable_events=True)
            ],
            [
                sg.Text('A: ', key=f'{SpatialFilter.name()}_HIGH_BOOST_A_TEXT', visible=False),
                sg.Input(size=(4, 1), justification='right', key=f'{SpatialFilter.name()}_HIGH_BOOST_A', visible=False)
            ],
            [
                sg.Text('Mask Size: ')
            ],
            [
                sg.Slider(range=(1, 51), default_value=9, size=(20,15), orientation='horizontal', resolution=2, key=f'{SpatialFilter.name()}_SIZE')
            ],
            [
                sg.Button('Apply', key=f'{SpatialFilter.name()}_APPLY'),
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Operations
    @staticmethod
    def scale_values(arr: npt.NDArray, max_value: int) -> npt.NDArray:
        negative_adjusted = arr - np.min(arr)
        scaled_down = negative_adjusted / np.max(negative_adjusted)
        scaled_up = max_value * scaled_down
        return scaled_up

    @staticmethod
    def scale_values_clip_bottom(arr: npt.NDArray, max_value: int) -> npt.NDArray:
        scaled_down = arr / np.max(arr)
        scaled_up = max_value * scaled_down
        tmp = np.clip(scaled_up, 0, max_value) 
        return tmp

    @staticmethod
    def convolve_filter(img: npt.NDArray, filt: npt.NDArray) -> npt.NDArray:
        convolve = lambda chunk: np.sum(chunk * filt)
        return SpatialFilter.apply_function(img, filt.shape[0], convolve)
    
    @staticmethod
    def apply_function(img: npt.NDArray, mask_size: int, func: callable) -> npt.NDArray:
        source_img = np.copy(img)
        Y, X = source_img.shape # Y, X
        
        result_img = np.empty(source_img.shape)
        
        half_mask = math.floor(mask_size / 2)
        
        p = half_mask
        source_img = np.pad(source_img, p)
    
        for y in range(Y):
            for x in range(X):
                # Find bounds of the local neighborhood
                x_start = x - half_mask
                x_end =   x + half_mask + 1
                y_start = y - half_mask
                y_end =   y + half_mask + 1

                chunk = source_img[y_start+p:y_end+p, x_start+p:x_end+p]

                result_img[y,x] = func(chunk)
        return result_img

    @staticmethod
    def box_filter(source_image: dict[str, str], mask_size: int) -> npt.NDArray:
        # Box filter
        data = np.copy(source_image['image'])
        box_filter = np.full((mask_size, mask_size), 1 / (mask_size**2))
        tmp = SpatialFilter.convolve_filter(data, box_filter)
        tmp = SpatialFilter.scale_values(tmp, 2**source_image['gray_resolution'] - 1)
        
        return tmp

    @staticmethod
    def smooth(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        # Box filter
        source_image['image'] = SpatialFilter.box_filter(source_image, mask_size).astype(np.uint8) 
        return source_image

    @staticmethod
    def median(modified: dict[str, str], mask_size: int) -> dict[str, str]:
        source_img = modified['image'] 
        Y, X = source_img.shape # Y, X
        result_img = np.empty(source_img.shape)

        half_mask = math.floor(mask_size / 2)

        for y in range(Y):
            for x in range(X):
                # Find bounds of the local neighborhood
                x_start = max(x - half_mask, 0)
                x_end =   min(x + half_mask + 1, X)
                y_start = max(y - half_mask, 0)
                y_end =   min(y + half_mask + 1, Y)

                chunk = source_img[y_start:y_end, x_start:x_end]

                result_img[y,x] = math.floor(np.median(chunk.flatten()))

        modified['image'] = result_img.astype(np.uint8) 
        return modified
    
    @staticmethod
    def laplacian(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        # Define Laplacian filter
        lap_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
        tmp = SpatialFilter.convolve_filter(source_image['image'], lap_filter)

        c = -1
        tmp = source_image['image'] + c * tmp

        # Laplacian potentially creates values outside of the image range, clip it to valid values only
        tmp = SpatialFilter.scale_values_clip_bottom(tmp, 2**source_image['gray_resolution'] - 1)
        print(np.min(tmp), np.max(tmp))

        source_image['image'] = tmp.astype(np.uint8)
        return source_image

    @staticmethod
    def highBoost(source_image: dict[str, str], mask_size: int, A: float) -> dict[str, str]:
        img = source_image['image']
        blurry = SpatialFilter.box_filter(source_image, mask_size)
        mask = img - blurry
        high_boosted = img + (A * mask)

        high_boosted = SpatialFilter.scale_values(high_boosted, 2**source_image['gray_resolution'] - 1)

        
        source_image['image'] = high_boosted.astype(np.uint8) 

        
        return source_image

    @staticmethod
    def apply(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        filter_type = values[f'{SpatialFilter.name()}_TYPE']
        filter_size = int(values[f'{SpatialFilter.name()}_SIZE'])

        if filter_type == 'Smoothing':
            return SpatialFilter.smooth(modified, filter_size)
        elif filter_type == 'Median':
            return SpatialFilter.median(modified, filter_size)
        elif filter_type == 'Laplacian':
            return SpatialFilter.laplacian(modified, filter_size)
        elif filter_type == 'High-Boost':
            return SpatialFilter.highBoost(modified, filter_size, float(values[f'{SpatialFilter.name()}_HIGH_BOOST_A']))
        else:
            print('ERROR: unknown type of filter requested')
            return modified

    @staticmethod
    def select_type(source_image: dict[str, str], working_image: dict[str, str], window, values) -> dict[str, str]:
        window[f'{SpatialFilter.name()}_HIGH_BOOST_A_TEXT'].update(visible=values[f'{SpatialFilter.name()}_TYPE'] == 'High-Boost')
        window[f'{SpatialFilter.name()}_HIGH_BOOST_A'].update(visible=values[f'{SpatialFilter.name()}_TYPE'] == 'High-Boost')
        return working_image

    # Events
    @staticmethod
    def get_operation(operation_name: str) -> callable:
        if operation_name == f'{SpatialFilter.name()}_TYPE':
            return SpatialFilter.select_type
        elif operation_name == f'{SpatialFilter.name()}_APPLY':
            return SpatialFilter.apply
        else:
            print("Unknown spatial filter operation")
        return

def get_instance():
    return SpatialFilter()
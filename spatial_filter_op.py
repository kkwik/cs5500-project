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
                sg.Combo(['Smoothing', 'Median', 'Maximum', 'Minimum', 'Midpoint', 'Laplacian', 'High-Boost', 'Arithmetic Mean', 'Geometric Mean', 'Harmonic Mean', 'Contraharmonic Mean', 'Alpha-Trimmed Mean'], default_value='Smoothing', key=f'{SpatialFilter.name()}_TYPE', enable_events=True)
            ],
            [
                sg.Text('A: ', key=f'{SpatialFilter.name()}_INPUT_LABEL', visible=False),
                sg.Input(size=(4, 1), justification='right', key=f'{SpatialFilter.name()}_INPUT_TEXT', visible=False),
                sg.Slider(range=(0, 8), default_value=1, size=(20,15), orientation='horizontal', key=f'{SpatialFilter.name()}_INPUT_SLIDER', visible=False)
            ],
            [
                sg.Text('Mask Size: ')
            ],
            [
                sg.Slider(range=(1, 51), default_value=3, size=(20,15), orientation='horizontal', resolution=2, key=f'{SpatialFilter.name()}_SIZE', enable_events=True)
            ],
            [
                sg.Button('Apply', key=f'{SpatialFilter.name()}_APPLY'),
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Helper Functions
    
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
    
    # Applies function on chunks of an image. Pad the image so that complete chunks are formed
    @staticmethod
    def apply_function_padded(img: npt.NDArray, mask_size: int, func: callable) -> npt.NDArray:
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

                # print(chunk, chunk.size, np.prod(chunk), np.prod(chunk)**(1/chunk.size))

                result_img[y,x] = func(chunk)
                # print(chunk, ' -> ', result_img[y,x])
        return result_img
    
    # Applies function on chunks of an image. Chunks overlapping borders will be cut short of full size
    @staticmethod
    def apply_function(img: npt.NDArray, mask_size: int, func: callable) -> npt.NDArray:
        source_img = img
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

                result_img[y,x] = func(chunk)

        return result_img

    # Convolve filter on image. Note: pads the image before applying the filter
    @staticmethod
    def convolve_filter(img: npt.NDArray, filt: npt.NDArray) -> npt.NDArray:
        convolve = lambda chunk: np.sum(chunk * filt)
        return SpatialFilter.apply_function_padded(img, filt.shape[0], convolve)

    @staticmethod
    def box_filter(source_image: dict[str, str], mask_size: int) -> npt.NDArray:
        # Box filter
        data = np.copy(source_image['image'])
        box_filter = np.full((mask_size, mask_size), 1 / (mask_size**2))
        tmp = SpatialFilter.convolve_filter(data, box_filter)
        tmp = SpatialFilter.scale_values(tmp, 2**source_image['gray_resolution'] - 1)
        
        return tmp

    # Filters

    @staticmethod
    def smooth(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        # Box filter
        source_image['image'] = SpatialFilter.box_filter(source_image, mask_size).astype(np.uint8) 
        return source_image

    @staticmethod
    def median(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        img = source_image['image']
        median = lambda chunk: math.floor(np.median(chunk))
        source_image['image'] = SpatialFilter.apply_function(img, mask_size, median).astype(np.uint8) 
        return source_image

    @staticmethod
    def maximum(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        img = source_image['image']
        max_f = lambda chunk: math.floor(np.max(chunk))
        source_image['image'] = SpatialFilter.apply_function(img, mask_size, max_f).astype(np.uint8) 
        return source_image

    @staticmethod
    def minimum(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        img = source_image['image']
        min_f = lambda chunk: math.floor(np.min(chunk))
        source_image['image'] = SpatialFilter.apply_function(img, mask_size, min_f).astype(np.uint8) 
        return source_image
    
    @staticmethod
    def midpoint(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        img = source_image['image']
        mid = lambda chunk: np.add(int(np.min(chunk)), int(np.max(chunk))) / 2
        source_image['image'] = SpatialFilter.apply_function(img, mask_size, mid).astype(np.uint8) 
        return source_image

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
    def arithmetic_mean(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        img = source_image['image']
        arith_mean = lambda chunk: np.sum(chunk) / (chunk.size)
        source_image['image'] = SpatialFilter.apply_function(img, mask_size, arith_mean).astype(np.uint8) 
        return source_image

    @staticmethod
    def geometric_mean(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        img = source_image['image']
        geo_mean = lambda chunk: np.exp(np.sum(np.log(chunk)) / chunk.size)
        tmp = SpatialFilter.apply_function(img, mask_size, geo_mean)
        source_image['image'] = SpatialFilter.scale_values(tmp, 255).astype(np.uint8) 
        return source_image

    @staticmethod
    def harmonic_mean(source_image: dict[str, str], mask_size: int) -> dict[str, str]:
        img = source_image['image']
        harm_mean = lambda chunk: (chunk.size) / np.sum(1/chunk)
        source_image['image'] = SpatialFilter.apply_function(img, mask_size, harm_mean).astype(np.uint8) 
        return source_image

    @staticmethod
    def contraharmonic_mean(source_image: dict[str, str], mask_size: int, Q: float) -> dict[str, str]:
        img = source_image['image']
        contraharm_mean = lambda chunk: np.sum(chunk**(Q+1)) / np.sum(chunk**Q)
        source_image['image'] = SpatialFilter.apply_function(img, mask_size, contraharm_mean).astype(np.uint8) 
        return source_image

    @staticmethod
    def alpha_trim_mean(source_image: dict[str, str], mask_size: int, d: int) -> dict[str, str]:
        img = source_image['image']
        alpha_trim = lambda chunk: np.sum(chunk) / ((chunk.size) - d)
        tmp = SpatialFilter.apply_function(img, mask_size, alpha_trim)
        tmp[0] = 0
        tmp[-1] = 0
        tmp[:,0] = 0
        tmp[:,-1] = 0
        source_image['image'] = SpatialFilter.scale_values(tmp, 255).astype(np.uint8)
        return source_image


    @staticmethod
    def apply(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        input_image = copy.deepcopy(modified)
        filter_type = values[f'{SpatialFilter.name()}_TYPE']
        filter_size = int(values[f'{SpatialFilter.name()}_SIZE'])

        if filter_type == 'Smoothing':
            return SpatialFilter.smooth(input_image, filter_size)
        elif filter_type == 'Median':
            return SpatialFilter.median(input_image, filter_size)
        elif filter_type == 'Laplacian':
            return SpatialFilter.laplacian(input_image, filter_size)
        elif filter_type == 'High-Boost':
            return SpatialFilter.highBoost(input_image, filter_size, float(values[f'{SpatialFilter.name()}_INPUT_TEXT']))
        elif filter_type == 'Arithmetic Mean':
            return SpatialFilter.arithmetic_mean(input_image, filter_size)
        elif filter_type == 'Geometric Mean':
            return SpatialFilter.geometric_mean(input_image, filter_size)
        elif filter_type == 'Harmonic Mean':
            return SpatialFilter.harmonic_mean(input_image, filter_size)
        elif filter_type == 'Contraharmonic Mean':
            return SpatialFilter.contraharmonic_mean(input_image, filter_size, float(values[f'{SpatialFilter.name()}_INPUT_TEXT']))
        elif filter_type == 'Alpha-Trimmed Mean':
            return SpatialFilter.alpha_trim_mean(input_image, filter_size, int(values[f'{SpatialFilter.name()}_INPUT_SLIDER']))
        elif filter_type == 'Minimum':
            return SpatialFilter.minimum(input_image, filter_size)
        elif filter_type == 'Maximum':
            return SpatialFilter.maximum(input_image, filter_size)
        elif filter_type == 'Midpoint':
            return SpatialFilter.midpoint(input_image, filter_size)
        else:
            print('ERROR: unknown type of filter requested')
            return input_image

    @staticmethod
    def select_type(source_image: dict[str, str], working_image: dict[str, str], window, values) -> dict[str, str]:
        selected_type = values[f'{SpatialFilter.name()}_TYPE']

        text_input_types = ['High-Boost', 'Contraharmonic Mean']
        slider_input_types = ['Alpha-Trimmed Mean']

        show_text_input = selected_type in text_input_types
        show_slider_input = selected_type in slider_input_types

        window[f'{SpatialFilter.name()}_INPUT_LABEL'].update(visible=show_text_input or show_slider_input)

        window[f'{SpatialFilter.name()}_INPUT_TEXT'].update(visible=show_text_input)
        window[f'{SpatialFilter.name()}_INPUT_SLIDER'].update(visible=show_slider_input)

        if selected_type == 'High-Boost':
            window[f'{SpatialFilter.name()}_INPUT_LABEL'].update('A: ')
        elif selected_type == 'Contraharmonic Mean':
            window[f'{SpatialFilter.name()}_INPUT_LABEL'].update('Q: ')
        elif selected_type == 'Alpha-Trimmed Mean':
            window[f'{SpatialFilter.name()}_INPUT_LABEL'].update('d: ')
        
        return working_image

    @staticmethod
    def mask_size_change(source_image: dict[str, str], working_image: dict[str, str], window, values) -> dict[str, str]:
        mn = int(values[f'{SpatialFilter.name()}_SIZE'])**2
        window[f'{SpatialFilter.name()}_INPUT_SLIDER'].update(range=(0, mn - 1))
        return working_image

    # Events
    @staticmethod
    def get_operation(operation_name: str) -> callable:
        if operation_name == f'{SpatialFilter.name()}_TYPE':
            return SpatialFilter.select_type
        elif operation_name == f'{SpatialFilter.name()}_SIZE':
            return SpatialFilter.mask_size_change
        elif operation_name == f'{SpatialFilter.name()}_APPLY':
            return SpatialFilter.apply
        else:
            print("Unknown spatial filter operation")
        return

def get_instance():
    return SpatialFilter()

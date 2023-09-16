import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np
from math import floor, ceil
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
                sg.Text('X: '),
                sg.Input(size=(4, 1), justification='right', key='ZOOM_X_RES', enable_events=True)
            ],
            [
                sg.Text('Y: '),
                sg.Input(size=(4, 1), justification='right', key='ZOOM_Y_RES', disabled=True)
            ],
            [
                sg.Checkbox('Lock ratio', key='ZOOM_LOCK_RATIO', default=True, enable_events=True)
            ],
            [
                sg.Button("Apply", key="ZOOM_APPLY"), 
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Operations

    @staticmethod
    def lock_ratio(source_image, working_image, window, values):
        window['ZOOM_Y_RES'].update(disabled=values['ZOOM_LOCK_RATIO'])
        return working_image

    # X input occured, if lock_ration == true then update text
    @staticmethod
    def x_input(source_image, working_image, window, values):
        if values['ZOOM_X_RES'] == '' or not values['ZOOM_X_RES'].isdigit(): # If there is not valid content in the X input to try to interpolate Y, empty Y text box
            window['ZOOM_Y_RES'].update('')
        elif values['ZOOM_LOCK_RATIO']:
            ratio = working_image['image'].shape[0] / working_image['image'].shape[1]
            window['ZOOM_Y_RES'].update(round(ratio * int(values['ZOOM_X_RES'])))
        return working_image # Modify ui without changing image

    @staticmethod
    def zoom_out(base_image, new_image):
        old_width = base_image.shape[1]
        new_width = new_image.shape[1]

        num_cols_to_remove = old_width - new_width

        removal_spacing = old_width / num_cols_to_remove

        del_cols = { int(removal_spacing * x) for x in range(ceil(num_cols_to_remove))} # Create list of columns to delete. Set to guarantee no duplicates
        del_cols = sorted(list(del_cols))

        # TODO: Check if necessary
        if num_cols_to_remove != len(del_cols):
            print("MISMATCH IN COLUMNS TO REMOVE: ", num_cols_to_remove, ' vs ', len(del_cols))
            diff = abs(num_cols_to_remove - len(del_cols))

            if len(del_cols) > num_cols_to_remove:
                del_cols = del_cols[:-diff]

        del_cols_indices = [n - i for i, n in enumerate(del_cols)] # Adjust indices for deletion of prior entries

        # Delete columns
        for del_col in del_cols_indices:
            base_image = np.delete(base_image, del_col, 1)

        return base_image
    
    @staticmethod
    def find_nearest(source, pixel, factor):
        source_mapped_pixel = np.array([pixel[0],np.floor(pixel[1] / factor).astype(int)])  # Map the pixel location in the upscaled image to the closest pixel in the source image
        return source[source_mapped_pixel[0]][source_mapped_pixel[1]]   # Retrieve the mapped pixel value

    @staticmethod
    def zoom_nn(base_image, new_image):
        old_dim = np.asarray(base_image.shape)
        new_dim = np.asarray(new_image.shape)

        old_width = base_image.shape[1]
        new_width = new_image.shape[1]

        factor = new_width / old_width

        res = np.zeros(new_image.shape)
        res[:] = -1 # Give a nonsense value so we know which values still need to be filled in

        # i == y, j == x
        # Transfer real values to new size
        # Interestingly, while this first set of for loops can be removed and the second one relied on, 
        # I've found it marginally faster to include this one. Presumably this is because the math for this 
        # one is less impactful than the math for the next one so doing work here is marginally more efficient
        # Difference is maybe 0.5 seconds on lena zooming in 200%
        for y in range(old_dim[0]):
            for x in range(old_dim[1]):
                res[y][round(factor * x)] = base_image[y][x]

        
        # Fill in gaps
        for y in range(new_dim[0]):
            for x in range(new_dim[1]):
                if res[y][x] == -1:
                    res[y][x] = Zoom.find_nearest(base_image, np.array([y,x]), factor)

        return res.astype(np.uint8) # Convert to uint8 at the end. Note: if there are any -1's leftover they will be wrapped to 255

    @staticmethod
    def find_flanking_values(row, pos):
        # In a numpy row, given a position, return the closest value to the left and right that are not -1
        left = reversed(row[:pos].tolist())
        left = [ i + 1 if n != -1 else -1 for i, n in enumerate(left)]
        left = np.array(left)
        left = np.setdiff1d(left, -1)
        left = left[0] if len(left) > 0 else 0
        left = -left
        
        right = row[pos + 1:].tolist()
        right = [ i + 1 if n != -1 else -1 for i, n in enumerate(right)]
        right = np.array(right)
        right = np.setdiff1d(right, -1)
        right = right[0] if len(right) > 0 else 0

        return (left, right)

    @staticmethod
    def zoom_linear(base_image, new_image):
        old_dim = np.asarray(base_image.shape)
        new_dim = np.asarray(new_image.shape)

        old_width = base_image.shape[1]
        new_width = new_image.shape[1]

        factor = new_width / old_width

        res = np.zeros(new_image.shape)
        res[:] = -1 # Give a nonsense value so we know which values still need to be filled in

        # i == y, j == x
        # Transfer real values to new size
        # Interestingly, while this first set of for loops can be removed and the second one relied on, 
        # I've found it marginally faster to include this one. Presumably this is because the math for this 
        # one is less impactful than the math for the next one so doing work here is marginally more efficient
        # Difference is maybe 0.5 seconds on lena zooming in 200%
        for y in range(old_dim[0]):
            for x in range(old_dim[1]):
                res[y][round(factor * x)] = base_image[y][x]

        
        # Fill in gaps
        for y in range(new_dim[0]):
            for x in range(new_dim[1]):
                if res[y][x] == -1:
                    row = res[y]
                    left, right = Zoom.find_flanking_values(row, x)

                    x0 = x + left
                    f0 = res[y][x0]
                    
                    x1 = x + right
                    f1 = res[y][x1]

                    f = ( f0 * (x1 - x) + f1 * (x - x0) ) / (x1 - x0)
                    
                    res[y][x] = f

        return res.astype(np.uint8) # Convert to uint8 at the end. Note: if there are any -1's leftover they will be wrapped to 255

    @staticmethod
    def zoom_bilinear(base_image, new_image):
        return

    @staticmethod
    def zoom(image, new_width, zoom_type):
        # Create empty image with new dimensions
        old_width = image.shape[1]
        new_image = np.zeros((image.shape[0], new_width))
        new_image[:] = -1

        if new_width == old_width:
            return image
        elif new_width > old_width:
            print('ZOOMING IN')
            if zoom_type == 'Nearest Neighbor':
                return Zoom.zoom_nn(image, new_image)
            elif zoom_type == 'Linear' or zoom_type == 'Bilinear':
                return Zoom.zoom_linear(image, new_image)
            elif zoom_type == 'Bilinear':
                return Zoom.zoom_bilinear(image, new_image)
            else:
                print('ERROR: unknown type of zoom requested')
                return image
        else:
            print('ZOOMING OUT')
            return Zoom.zoom_out(image, new_image)

    


    @staticmethod
    def zoom_apply(source_image, working_image, window, values):
        print("ZOOM APPLY")

        new_width = int(values['ZOOM_X_RES'])
        new_height = int(values['ZOOM_Y_RES'])

        zoom_type = values['ZOOM_TYPE']
        image = np.copy(working_image['image']) # Figure out if copy is actually necessary
        
        # Apply on x direction
        # print('Applying zoom on x ', image.shape)
        image = Zoom.zoom(image, new_width, zoom_type)
        # print('Applied zoom on x ', image.shape)
        
        # Apply on y direction
        # print('Applying zoom on y ', image.shape)
        image = np.rot90(image)  # Rotate image. Zoom() will apply on x axis only, we'll rotate and then just apply again to change y axis
        image = Zoom.zoom(image, new_height, zoom_type)
        image = np.rot90(image, 3)   # Rotate back

        # print('Applied zoom on y', image.shape)

        working_image['image'] = image
        return working_image

    # Events
    @staticmethod
    def get_operation(operation_name):
        
        if operation_name == f'{Zoom.name()}_APPLY':
            return Zoom.zoom_apply
        elif operation_name == 'ZOOM_LOCK_RATIO':
            return Zoom.lock_ratio
        elif operation_name == 'ZOOM_X_RES':
            return Zoom.x_input
        else:
            print("Unknown zoom operation")
        return

def get_instance():
    return Zoom()

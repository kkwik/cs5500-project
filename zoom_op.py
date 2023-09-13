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
                sg.Text('X: '),
                sg.Input(size=(4, 1), justification='right', key='ZOOM_X_RES')
            ],
            [
                sg.Text('Y: '),
                sg.Input(size=(4, 1), justification='right', key='ZOOM_Y_RES')
            ],
            [
                sg.Checkbox('Lock ratio', key='ZOOM_LOCK_RATIO', enable_events=True)
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

    @staticmethod
    def zoom_out(start_image, factor):
        old_dim = np.asarray(start_image.shape)
        new_dim = (factor * old_dim).round().astype(np.int64)

        spacing = 1 / factor
        

        # Get indexes of rows and cols to delete
        del_rows =  list(filter(lambda num: num % spacing == 0, range(old_dim[0])))
        del_cols = list(filter(lambda num: num % spacing == 0, range(old_dim[1])))

        # print(list(range(old_dim[0])))
        print(del_rows)
        
        # Adjust indices for deletion of prior entries
        del_cols = [n - i for i, n in enumerate(del_cols)]
        del_rows = [n - i for i, n in enumerate(del_rows)]


        for del_col in del_cols:
            start_image = np.delete(start_image, del_col, 1)
        
        for del_row in del_rows:
            start_image = np.delete(start_image, del_row, 0)
            
        print(start_image.shape)
        return start_image
    
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

        if factor > 1.0:
        
            # i == y, j == x
            # Transfer real values to new size
            # Interestingly, while this first set of for loops can be removed and the second one relied on, 
            # I've found it marginally faster to include this one. Presumably this is because the math for this 
            # one is less impactful than the math for the next one so doing work here is marginally more efficient
            # Difference is maybe 0.5 seconds on lena zooming in 200%
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
    def zoom_apply(source_image, working_image, window, values):
        print("ZOOM APPLY")
        print(values)


        factor = values['ZOOM_SLIDER'] / 100
        result_image = working_image['image']

        if values['ZOOM_SLIDER'] == 100:
            return working_image # Requested zoom is 100%, exit
        elif factor < 1:
            result_image = Zoom.zoom_out(working_image['image'], factor)
        else:
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
        elif operation_name == 'ZOOM_LOCK_RATIO':
            return Zoom.lock_ratio
        else:
            print("Unknown zoom operation")
        return

def get_instance():
    return Zoom()

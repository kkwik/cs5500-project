import PySimpleGUI as sg
import PIL
from PIL import Image
import util
import importlib

import numpy as np
import os 


# Import operations
op_files = [ filename for filename in os.listdir() if filename.endswith('_op.py') ] # List containing the names of all suspected image operation files
ops = {}   # List of loaded image operations

# Attempt importing detected image operations
for op_file in op_files:
    try:
        module = importlib.import_module(op_file.removesuffix('.py'))
        op = module.get_instance()
        
        if op.name() in ops.keys():
            raise ImportError(f'An image operation already exists with the name {op.name()}')
            continue
        
        ops[op.name()] = op
    except ImportError as err:
        print(f'Error: failed to import {op_file}: {err}')

ops = ops.values()

# Define layout
input_selection_row = [
    [
        sg.Text("Select Input Image", key="input_text"),
        sg.In(size=(30,1), enable_events=True, key="IMAGE_SELECTED"),
        sg.FileBrowse()
    ],
]

input_column = [
    [
        sg.Text("Input Image")
    ],
    [
        sg.Image(key="INPUT_DISPLAY")
    ],
    [
        sg.Text('', key='INPUT_RES')
    ]
]

operations_column = [
        [sg.Frame(op.name().title(), [[op.get_gui()]], key=f'{op.name()}_FRAME')] for op in ops
    ]

output_column = [
    [
        sg.Text("Output Image")
    ],
    [
        sg.Image(key="OUTPUT_DISPLAY")
    ],
    [
        sg.Text('', key='OUTPUT_RES')
    ]
]

layout = [
    [
        sg.Column(input_selection_row),
    ],
    [
        sg.Column(input_column),
        sg.VSeperator(),
        sg.Column(operations_column),
        sg.VSeperator(),
        sg.Column(output_column)
    ]
]

window = sg.Window(title="Program", layout=layout, resizable=True, element_justification='c')

original_data = {}
modified_data = {}

def update_image(sgImage, pixel_array):
    sgImage.update(data=util.np_arr_to_byteio(pixel_array))

def update_img_res(sgText, pixel_array):
    sgText.update(f'{pixel_array.shape[1]}x{pixel_array.shape[0]}')

while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "IMAGE_SELECTED":
        # Input Image updated, update display

        filename = values["IMAGE_SELECTED"]
        
        input_image = Image.open(filename)      # Load image
        input_image = input_image.convert('L') # Convert to 8-bit grayscale
        original_data['image'] = np.asarray(input_image) # Extract pixel values as array
        original_data['gray_resolution'] = 8
        modified_data = original_data.copy()              # Set initial modified_data

        update_image(window['INPUT_DISPLAY'], original_data['image']) # Update input image display
        update_img_res(window['INPUT_RES'], original_data['image'])
        update_image(window['OUTPUT_DISPLAY'], modified_data['image']) # Update output image display
        update_img_res(window['OUTPUT_RES'], modified_data['image'])


    elif values['IMAGE_SELECTED'] != '' and any(event.startswith(op.name()) for op in ops):
        image_op_class = list(filter(lambda op: event.startswith(op.name()), ops))[0] # This access is safe because on import we check so ops contains only unique operation names. Thus any() returning true means exactly one matches exist
        image_operation = image_op_class.get_operation(operation_name=event)
        
        related_values = { key: values[key] for key in values if key.startswith(image_op_class.name()) }    # We want to give the image operator all relevant information to do it's job, so give it all values that start with the name of the operation class
        modified_data = image_operation(original_data, modified_data, window, related_values)

        update_image(window['OUTPUT_DISPLAY'], modified_data['image']) # Update output image display
        update_img_res(window['OUTPUT_RES'], modified_data['image'])
    else:
        print('Error: failed to find corresponding action for event')
        print('Event:', event)
        print('Values:', values)
        print()
            


window.close()
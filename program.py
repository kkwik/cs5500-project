import PySimpleGUI as sg
import PIL
from PIL import Image
from util import convert_to_bytes
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
    ]
]

operations_column = [
    [
        sg.Frame(op.name().title(), [[op.get_gui()]], key=f'{op.name()}_FRAME') for op in ops
    ]
]

output_column = [
    [
        sg.Text("Output Image")
    ],
    [
        sg.Image(key="OUTPUT_DISPLAY")
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

input_image = []

while True:
    event, values = window.read()
    
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "IMAGE_SELECTED":
        # Input Image updated, update display
        try:
            filename = values["IMAGE_SELECTED"]
            input_image = convert_to_bytes(filename)
            window["INPUT_DISPLAY"].update(data=input_image)
            
        except:
            pass

        print(np.array(Image.open(filename)))
    elif any(event.startswith(op.name()) for op in ops):
        # On import ops is verified to have unique name values so any() returning true means 1 matching
        operation = list(filter(lambda op: event.startswith(op.name()), ops))[0].get_operation(event)
        operation('')
    else:
        print('Error: failed to find corresponding action for event')
        print("Event:", event)
        print("Values:", values)
        print()
            


window.close()

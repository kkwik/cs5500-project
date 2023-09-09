import PySimpleGUI as sg
import PIL
from PIL import Image
from util import convert_to_bytes
import zoom
import numpy as np

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
        sg.Frame("Zoom", [[zoom.get_gui()]], key="ZOOM_FRAME")
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
    elif event.startswith("ZOOM"):
        zoom.handle_event(event, input_image)
    else:
        print("Event:", event)
        print("Values:", values)
            


window.close()

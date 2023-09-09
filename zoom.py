import PySimpleGUI as sg

def get_gui():
    contents = [
        [
            sg.Text("Zoom Operation"),
        ],
        [
            sg.Button("+", key="ZOOM_IN"), 
            sg.Button("-", key="ZOOM_OUT")
        ]
    ]

    gui = sg.Column(contents)

    return gui

# Events
def handle_event(operation, image):
    
    if operation == "ZOOM_IN":
        print("Zoom in")
    elif operation == "ZOOM_OUT":
        print("Zoom out")
    return


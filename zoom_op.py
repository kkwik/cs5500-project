import PySimpleGUI as sg

# Name of the operation
name = "ZOOM"

# Return the UI elements of the operation
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

# Operations
def zoom_in(input):
    print("ZOOM IN")
    return ""

def zoom_out(input):
    print("ZOOM OUT")
    return ""

# Events
def get_operation(operation_name):
    
    if operation_name == "ZOOM_IN":
        return zoom_in
    elif operation_name == "ZOOM_OUT":
        return zoom_out
    else:
        print("Unknown zoom operation")
    return


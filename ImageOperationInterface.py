import PySimpleGUI as sg

class ImageOperationInterface:
    def name(self) -> str:
        """Return a string identifying the name of the image operation type."""
        pass

    def get_gui(self) -> sg.Column:
        """Return the gui for the image operation type."""
        pass

    def get_operation(self, operation_subtype: str):
        """Return the function that implements the desired image operation subtype."""
        pass

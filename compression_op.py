import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


class Compression(ImageOperationInterface):
    @staticmethod
    def name():
        return 'COMPRESSION'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Combo(['Grayscale RLE', 'Bitplane RLE', 'Huffman'], default_value='Grayscale RLE', key=f'{Compression.name()}_SAVE_TYPE'),
                sg.Text("File Name:"),
                sg.Input('compressed_file', size=(10,1), key=f'{Compression.name()}_FILE_NAME'),
                sg.Button('Save', key=f'{Compression.name()}_SAVE')
            ],
            [
                sg.Text("Load Image"),
                sg.In(size=(30,1), enable_events=True, key=f'{Compression.name()}_IMAGE_SELECTED'),
                sg.FileBrowse()
            ]
        ]

        gui = sg.Column(contents)

        return gui

    # Operations
    ###
    # Compress
    ###
    @staticmethod
    def save_to_file(data: bytes, filename: str):
        with open(filename, 'wb') as file:
            file.write(data)

    @staticmethod
    def compress_grayscale(modified: dict[str, str]):
        image_data = modified['image']
        bit_depth = modified['gray_resolution']
        output = []
        output.append(0)

        # 00: Line end
        # 01: Image end

        prev_pixel_val = -1
        count = 0

        for y in range(plane.shape[0]):
            for x in range(plane.shape[1]):
                pixel = plane[y, x]

                if x == 0:
                    prev_pixel_val = pixel

                if pixel == prev_pixel_val:
                    if count == 255:
                        output.append(count)
                        output.append(pixel)
                        count = 0
                    count += 1

                else:
                    output.append(count)
                    output.append(prev_pixel_val)
                    count = 0
                    count += 1
                    prev_pixel_val = pixel
            output.append(count)
            output.append(prev_pixel_val)
            count = 0
            output.append(0)
            output.append(0)
        output.append(0)
        output.append(1)

        Compression.save_to_file(bytes(output), filename)

        return modified

    @staticmethod
    def compress_bitplane(modified: dict[str, str], filename):
        image_data = modified['image']
        bit_depth = modified['gray_resolution']
        output = []
        output.append(1) # First byte will indicate bitplane encoding
        # 000: End of line
        # 001: End of plane
        # 002: End image


        for d in range(bit_depth):
            plane = np.bitwise_and(np.right_shift(image_data, d), 1)

            prev_pixel_val = 1
            count = 0
            for y in range(plane.shape[0]):
                for x in range(plane.shape[1]):
                    pixel = plane[y, x]

                    # Handle case where start is a 0
                    if x == 0 and pixel != prev_pixel_val:
                        output.append(0)
                        prev_pixel_val = pixel

                    if pixel == prev_pixel_val:
                        if count == 255:
                            output.append(count)
                            output.append(0)
                            count = 0
                        count += 1
                            
                    else:
                        output.append(count)
                        count = 0
                        count += 1
                        prev_pixel_val = pixel
                output.append(count)
                count = 0
                prev_pixel_val = 1
                output.append(0)
                output.append(0)
                output.append(0)
            output.append(0)
            output.append(0)
            output.append(1)

        Compression.save_to_file(bytes(output), filename)

        return modified

    @staticmethod
    def compress_huffman(modified: dict[str, str]):

        return modified

    @staticmethod
    def save(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        save_type = values[f'{Compression.name()}_SAVE_TYPE']

        if save_type == "Grayscale RLE":
            Compression.compress_grayscale(modified, values[f'{Compression.name()}_FILE_NAME'])
        elif save_type == "Bitplane RLE":
            Compression.compress_bitplane(modified, values[f'{Compression.name()}_FILE_NAME'])
        elif save_type == "Huffman":
            Compression.compress_huffman(modified, values[f'{Compression.name()}_FILE_NAME'])

        return modified

    ###
    # Decompress
    ###
    @staticmethod
    def decompress_grayscale(data: list[int]) -> npt.NDArray:
        return

    @staticmethod
    def decompress_bitplane(data: list[int]) -> npt.NDArray:
        val = 1
        output = 0
        plane = 0
        plane_count = 0
        row = []
        row_count = 0
        print(data)
        entries_left = len(data)
        with tqdm(total=entries_left) as pbar:
            while len(data) > 0:
                if len(data) > 2 and data[0] == 0 and data[1] == 0:
                    if data[2] == 0:
                        # new line
                        row = np.array(row)
                        if row_count == 0:
                            plane = row
                        else:
                            plane = np.vstack([plane, row])
    
                        row = []
                        row_count += 1
                        
                        
                    elif data[2] == 1:
                        # end plane
                        if plane_count == 0:
                            output = plane
                        else:
                            output = np.bitwise_or(output, np.left_shift(plane, plane_count))

                        plane_count += 1
                        row_count = 0

                    data = data[3:]
                    val = 1
                else:
                    
                    count = data[0]
                    row.extend(count * [val])
                    val = (val + 1) % 2
                    data = data[1:]
                
                entries_removed = entries_left - len(data)
                pbar.update(entries_removed)
                entries_left -= entries_removed
        print(output)
        return output.astype(np.uint8) 

    @staticmethod
    def decompress_huffman(data: list[int]) -> npt.NDArray:
        return

    @staticmethod
    def read_compressed_data(data: list[int]) -> npt.NDArray:
        type_indicator = data[0]
        data = data[1:]
        data_array = 0

        if type_indicator == 0:
            data_array = Compression.decompress_grayscale(data)
        elif type_indicator == 1:
            data_array = Compression.decompress_bitplane(data)
        elif type_indicator == 2:
            data_array = Compression.decompress_huffman(data)
        
        return data_array

    @staticmethod
    def compressed_file_selected(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        filename = values[f'{Compression.name()}_IMAGE_SELECTED']

        data = 0
        with open(filename, mode='rb') as file:
            data = file.read()
            data = [b for b in data]

        loaded_file = {}
        loaded_file['gray_resolution'] = 8
        loaded_file['image'] = Compression.read_compressed_data(data)
        return loaded_file




    # Events
    @staticmethod
    def get_operation(operation_name: str) -> callable:
        
        if operation_name == f'{Compression.name()}_SAVE':
            return Compression.save
        elif operation_name == f'{Compression.name()}_IMAGE_SELECTED':
            return Compression.compressed_file_selected
        else:
            print("Unknown gray operation")
        return

def get_instance():
    return Compression()

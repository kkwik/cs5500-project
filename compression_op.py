import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import pickle


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
    def save_to_file(data: bytes, filename: str, append: bool = False):
        mode = 'b'
        mode += 'a' if append else 'w'
        with open(filename, mode=mode) as file:
            file.write(data)

    @staticmethod
    def compress_grayscale(modified: dict[str, str], filename):
        image_data = modified['image']

        output = []
        output.append(0)    # First byte indicates type of compression. 0 is grayscale RLE

        # 00: Line end
        end_of_line = [0, 0]

        prev_pixel_val = -1
        count = 0

        for y in range(image_data.shape[0]):
            for x in range(image_data.shape[1]):
                pixel = image_data[y, x]

                if x == 0:
                    prev_pixel_val = pixel

                if pixel == prev_pixel_val:
                    # Handle case where count exceeds byte size
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
            output.extend(end_of_line)

        Compression.save_to_file(bytes(output), filename)

        return modified

    @staticmethod
    def compress_bitplane(modified: dict[str, str], filename):
        image_data = modified['image']
        bit_depth = modified['gray_resolution']
        output = []
        output.append(1) # First byte indicates type of compression. 1 is bitplane RLE

        # Signals
        # 000: End of line
        end_of_line = [0, 0, 0]
        # 001: End of plane
        end_of_plane = [0, 0, 1]


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
                        # Handle case where count exceeds byte size
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
                output.extend(end_of_line)
            output.extend(end_of_plane)

        Compression.save_to_file(bytes(output), filename)

        return modified

    # Recursively traverse binary tree and return dictionary mapping of value -> bitstring
    @staticmethod
    def recursive(input_tuple, bit_string: str) -> dict:
        if type(input_tuple) is not tuple:
            base_case = {}
            base_case[input_tuple] = bit_string

            return base_case
        
        left = input_tuple[0]
        right = input_tuple[1]

        result = {}
        result.update(Compression.recursive(left, bit_string + '0'))
        result.update(Compression.recursive(right, bit_string + '1'))

        return result

    @staticmethod
    def compress_huffman(modified: dict[str, str], filename):

        # File format
        # Byte 0: 2 - compression type indicator
        # 2 bytes for Codes size: 
        # Huffman Codes:
        # Data:
        
        image_data = modified['image']

        output = []
        output.append(2) # First byte indicates type of compression. 1 is bitplane RLE
        Compression.save_to_file(bytes(output), filename) # Save compression type

        ######################
        # Create Huffman Codes
        ######################
        probs = dict(zip(*np.unique(image_data, return_counts=True)))
        line_end = {'eol': image_data.shape[0] - 1}
        probs.update(line_end)

        # Calculate binary tree
        while len(probs) > 1:
            counts = sorted(probs.values())
            least_common_freq = counts[0]
            least_common_key = list(probs.keys())[list(probs.values()).index(least_common_freq)]
            least_common = probs[least_common_key]

            probs.pop(least_common_key)
            counts = counts[1:]

            second_least_common_freq = counts[0]
            second_least_common_key = list(probs.keys())[list(probs.values()).index(second_least_common_freq)]
            second_least_common = probs[second_least_common_key]

            probs.pop(second_least_common_key)
            counts = counts[1:]

            probs[(least_common_key, second_least_common_key)] = least_common_freq + second_least_common_freq

        # Tabulate huffman codes based on tree
        tree = list(probs.keys())[0]
        codes = Compression.recursive(tree, '')
        encoded_codes = pickle.dumps(codes)

        code_size = len(encoded_codes)

        # Sanity check on size of huffman code
        if code_size > 2*16:
            print(f'WARNING: Huffman code table size of {code_size} exceeds 2 byte limit. Failed to compress image')
            return modified

        binary_code_size = '{0:b}'.format(code_size)
        binary_code_size = ((len(binary_code_size) % 8) * '0') + binary_code_size # Left pad
        binary_code_size = [binary_code_size[i:i+8] for i in range(int(len(binary_code_size)/8))] # Split into bytes
        binary_code_size = [int(i, 2) for i in binary_code_size] # Interpret 8 bit strings as ints

        Compression.save_to_file(bytes(binary_code_size), filename, append=True) # Save codes size as 2 bytes

        # Save Huffman codes to file
        Compression.save_to_file(encoded_codes, filename, append=True)

        #####################
        # Encode pixel values
        #####################
        output = ''
        
        for y in range(image_data.shape[0]):
            for x in range(image_data.shape[1]):
                pixel = image_data[y, x]

                # If this is the first pixel of a new line, add eol signifier
                if x == 0:
                    output += codes['eol']

                output += codes[pixel]
        
        output += ((len(output) % 8) * '0') # Pad right side of data to 8 byte boundary
        output = [output[i:i+8] for i in range(int(len(output)/8))] # Split string on 8 bit boundaries
        output = [int(i, 2) for i in output] # Interpret 8 bit strings as ints



        encoded_data = bytes(output)

        Compression.save_to_file(encoded_data, filename, append=True)

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
        output = 0
        row = []
        row_count = 0

        entries_left = len(data)
        with tqdm(total=entries_left) as pbar:
            while len(data) > 0:
                if data[0] == 0 and data[1] == 0:
                    # new line
                    row = np.array(row)
                    if row_count == 0:
                        output = row
                    else:
                        output = np.vstack([output, row])
                    
                    row = []
                    row_count += 1

                    data = data[2:]
                else:
                    count = data[0]
                    pixel_value = data[1]
                    row.extend(count * [pixel_value])
                    data = data[2:]

                entries_removed = entries_left - len(data)
                pbar.update(entries_removed)
                entries_left -= entries_removed
        
        return output.astype(np.uint8) 

    @staticmethod
    def decompress_bitplane(data: list[int]) -> npt.NDArray:
        val = 1
        output = 0
        plane = 0
        plane_count = 0
        row = []
        row_count = 0

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

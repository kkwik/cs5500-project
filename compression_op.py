import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import pickle
from math import log


class Compression(ImageOperationInterface):
    LZW_BIT_LENGTH = 12

    @staticmethod
    def name():
        return 'COMPRESSION'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Combo(['Grayscale RLE', 'Bitplane RLE', 'Huffman', 'LZW'], default_value='Grayscale RLE', key=f'{Compression.name()}_SAVE_TYPE'),
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
    @staticmethod
    def intToBinaryString(num: int) -> str:
        return '{0:b}'.format(num)

    @staticmethod
    def leftPadBinaryString(bits: str, desiredBitCount: int = 0) -> str:
        if desiredBitCount == 0:
            return ((8 - (len(bits) % 8)) * '0') + bits if len(bits) % 8 != 0 else bits
        else:
            return (desiredBitCount - len(bits)) * '0' + bits if desiredBitCount > len(bits) else bits

    @staticmethod
    def splitBitStringIntoChunks(bits: str, chunk_size: int = 8) -> list[str]:
        return [bits[i*chunk_size:i*chunk_size+chunk_size] for i in range(int(len(bits)/chunk_size))]

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
        output.append(2) # First byte indicates type of compression. 2 indicates huffman
        Compression.save_to_file(bytes(output), filename) # Save compression type

        ######################
        # Create Huffman Codes
        ######################
        probs = dict(zip(*np.unique(image_data, return_counts=True)))
        # print(probs)
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
        # print(tree)
        codes = Compression.recursive(tree, '')


        #####################
        # Encode pixel values
        #####################
        data = ''
        
        for y in range(image_data.shape[0]):
            for x in range(image_data.shape[1]):
                pixel = image_data[y, x]

                # If this is the first pixel of a new line, add eol signifier
                if x == 0 and y != 0:
                    data += codes['eol']

                data += codes[pixel]
        data += codes['eol']
        # print(output)

        codes = {v: k for k, v in codes.items()} # During decompression this mapping needs to be reversed, so we do it during compression
        encoded_codes = pickle.dumps(codes)
        encoded_body = pickle.dumps({'table': codes, 'data': data})

        code_size = len(encoded_codes)

        if code_size > 2**16:
            print(f'WARNING: Huffman code table size of {code_size} exceeds 2 byte limit. Failed to compress image')
            return modified

        code_size = Compression.intToBinaryString(code_size)
        code_size = Compression.leftPadBinaryString(code_size)
        code_size = Compression.splitBitStringIntoChunks(code_size)
        code_size = [int(i, 2) for i in code_size]

        Compression.save_to_file(bytes(code_size), filename, append=True) # Append size of code table

        Compression.save_to_file(encoded_codes, filename, append=True) # Append code table

        encoded_data = '1' + data
        encoded_data = Compression.leftPadBinaryString(encoded_data)
        
        encoded_data = Compression.splitBitStringIntoChunks(encoded_data)
        encoded_data = [int(i, 2) for i in encoded_data]
        
        Compression.save_to_file(bytes(encoded_data), filename, append=True)

        return modified

    @staticmethod
    def get_lzw_dictionary(decoding: bool = False) -> dict:
        if not decoding:
            dictionary = {str(i): i for i in range(259)}
        else:
            dictionary = {i: str(i) for i in range(259)}
        # 256 will be clear table
        # 257  will be end of image
        # 258 will be end of line
        return dictionary
    
    @staticmethod
    def compress_lzw(modified: dict[str, str], filename):
        image_data = modified['image']

        output = []
        output.append(3) # First byte indicates type of compression. 3 indicates LZW
        Compression.save_to_file(bytes(output), filename) # Save compression type

        # Create initial dictionary
        dictionary = Compression.get_lzw_dictionary()

        # Insert eol indicators
        flattened = image_data.flatten().astype(np.uint)    # Allow larger values so we can insert the line end values which are > 255
        eol_indices = [image_data.shape[1] * (i + 1) for i in range(image_data.shape[0])]
        data = np.insert(flattened, eol_indices, 258)
        data = list(data)
        print(data)

        indices = []

        dictionary_size = 2**Compression.LZW_BIT_LENGTH

        normalize = lambda x: '-'.join([str(i) for i in x]) if type(x) is list else str(x)

        entries_left = len(data)
        with tqdm(total=entries_left) as pbar:
            while len(data) > 0:
                # Table size exceeded, clear the table
                if len(dictionary) >= dictionary_size:
                    indices.append(256)
                    dictionary = Compression.get_lzw_dictionary()

                keys = list(dictionary.keys()) # Get updated list of dictionary entries

                window = [data[0]]
                while normalize(window) in keys and len(window) < len(data):
                    window = data[:len(window) + 1]
                
                # print(f'{normalize(window)} not in dictionary, adding it. emitting {normalize(window[:-1])}, data: {data[:5]}')
                emit = window[:-1]

                if normalize(window) in keys:
                    # We've exited the above loop but the window is a key, 
                    # therefore we've run out of symbols to consume. Append what we have
                    indices.append(dictionary[normalize(window)])
                    data = data[len(window):]
                else:
                    indices.append(dictionary[normalize(emit)])
                    data = data[len(emit):]
                    dictionary[normalize(window)] = len(dictionary)
                
                entries_removed = entries_left - len(data)
                pbar.update(entries_removed)
                entries_left -= entries_removed

        #########
        # Storing
        #########
        # print(indices)
        indices = [Compression.leftPadBinaryString(Compression.intToBinaryString(n), Compression.LZW_BIT_LENGTH) for n in indices] # Turn integer indices into bitstrings and left pad as reqiured
        indices = ''.join(indices) # Combine all indices as one bit string
        indices = Compression.leftPadBinaryString(indices) # Because we're about to split on 8 bits we need to pad to 8 bits
        indices = Compression.splitBitStringIntoChunks(indices) # Split up into bytes
        indices = [int(i, 2) for i in indices]  # Convert back to ints

        Compression.save_to_file(bytes(indices), filename, append=True)

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
        elif save_type == "LZW":
            Compression.compress_lzw(modified, values[f'{Compression.name()}_FILE_NAME'])

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

        code_size = data[:2]
        data = data[2:]

        code_size = [Compression.intToBinaryString(b) for b in code_size]
        code_size = [Compression.leftPadBinaryString(b) for b in code_size]
        code_size = ''.join(code_size)
        code_size = int(code_size, 2)

        codes = data[:code_size]
        data = data[code_size:]
        codes = pickle.loads(bytes(codes))

        data = [Compression.intToBinaryString(b) for b in data]
        data = [Compression.leftPadBinaryString(b) for b in data]
        data = ''.join(data)
        
        data = data[data.index('1') + 1:]


        # Prepare Codes
        
        keys = list(codes.keys())

        output = 0
        row = []
        row_count = 0

        entries_left = len(data)
        with tqdm(total=entries_left) as pbar:
            while len(data) > 0:
                window = data[0]
                while window not in keys:
                    window = data[:len(window) + 1]

                data = data[len(window):]
                symbol = codes[window]

                if symbol == 'eol':
                        # new line
                        row = np.array(row)
                        if row_count == 0:
                            output = row
                        else:
                            output = np.vstack([output, row])
                        
                        row = []
                        row_count += 1
                else:
                    row.append(symbol)

                entries_removed = entries_left - len(data)
                pbar.update(entries_removed)
                entries_left -= entries_removed

        return output.astype(np.uint8) 
    
    @staticmethod
    def decompress_lzw(data: list[int]) -> npt.NDArray:
        # Read data back into 12 bit entries
        indices = [Compression.leftPadBinaryString(Compression.intToBinaryString(i)) for i in data] # Turn ints into bytes
        indices = ''.join(indices)  # Join bytes together
        indices = indices[len(indices) % 12:] # We need to remove any padding that was added to make the data fit /8
        indices = Compression.splitBitStringIntoChunks(indices, Compression.LZW_BIT_LENGTH) # Split bytes on 12 bit boundaries
        indices = [int(i, 2) for i in indices]
        # print(indices)

        # Decode data
        dictionary = Compression.get_lzw_dictionary(decoding=True)

        normalize = lambda x: '-'.join([str(i) for i in x]) if type(x) is list else str(x)

        symbols = []

        entries_left = len(indices)
        with tqdm(total=entries_left) as pbar:
            prev = ''
            while len(indices) > 0:
                keys = list(dictionary.keys()) # Get updated list of dictionary entries

                symbol = indices[0]
                indices = indices[1:]

                if symbol == 256:
                    dictionary = Compression.get_lzw_dictionary(decoding=True) # Clear dictionary
                elif symbol in keys:
                    result = dictionary[symbol]
                    symbols.extend(result.split('-'))
                    if prev:
                        dictionary[len(dictionary)] = normalize([prev, result.split('-')[0]])
                    prev = result
                else:
                    print('not in dict')
                    prev = normalize([prev, prev.split('-')[0]])
                    dictionary[len(dictionary)] = prev
                    symbols.extend(prev.split('-'))
                
                
                entries_removed = entries_left - len(indices)
                pbar.update(entries_removed)
                entries_left -= entries_removed

        symbols = [int(o) for o in symbols]
        
        # Convert 1D data back into 2D
        print(symbols)
        output = []
        row = []
        

        for s in symbols:
            if s == 258:
                # print(f'row: {len(row)}')
                output.append(row)
                row = []
            else:
                row.append(s)
        
        output = np.array(output)

        return output.astype(np.uint8)

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
        elif type_indicator == 3:
            data_array = Compression.decompress_lzw(data)
        
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

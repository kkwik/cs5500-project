import PySimpleGUI as sg
from ImageOperationInterface import ImageOperationInterface
import copy
import hashlib
from PIL import Image
import numpy as np
import util
import numpy.typing as npt


class Watermark(ImageOperationInterface):
    watermark = np.zeros(1)

    @staticmethod
    def name():
        return 'WATERMARK'

    # Return the UI elements of the operation
    @staticmethod
    def get_gui():
        contents = [
            [
                sg.Text("Select Watermark Image"),
                sg.In(size=(30,1), enable_events=True, key="WATERMARK_SELECTED"),
                sg.FileBrowse()
            ],
            [
                sg.Text('Image ID: '),
                sg.Input(justification='right', key=f'{Watermark.name()}_IMAGE_ID'),
            ],
            [
                sg.Text('User Key: '),
                sg.Input(justification='right', key=f'{Watermark.name()}_USER_KEY')
            ],
            [
                sg.Button('Insert', key=f'{Watermark.name()}_INSERT', disabled=True),
                sg.Button('Extract', key=f'{Watermark.name()}_EXTRACT', disabled=True)
            ]
        ]

        gui = sg.Column(contents)

        return gui

    def getMd5(input: str) -> str:
        return hashlib.md5(input.encode('utf-8')).hexdigest()

    def getHash(*args) -> str:
        hashInput = ''.join(map(str, args))
        hash = Watermark.getMd5(hashInput)
        return hash

    def hexStringToBitString(hash: str) -> str:
        return f'{int(hash, 16):0128b}'

    def getHashBlock(*args, desiredBlockSize: int, watermarkShape: tuple) -> npt.NDArray:
        hashString = Watermark.hexStringToBitString(Watermark.getHash(args))
        
        # Continually add to hashblock until it exceeds the desired size. Calculate subsequent hashes based on previous hash to avoid repeated values
        while len(hashString) < desiredBlockSize:
            hashString += Watermark.hexStringToBitString(Watermark.getHash(hashString))

        hashString = hashString[:desiredBlockSize] # Resize to cut off excess values
        hashBlock = np.array(list(hashString), dtype=np.uint8)
        hashBlock = np.array(np.split(hashBlock, watermarkShape[1]))    # Make rows
        return hashBlock
    
    def setArrayLSBTo(arr: npt.NDArray, val: bool) -> npt.NDArray:
        v = 1 if val else 0
        ret = np.zeros(arr.shape, dtype=arr.dtype)
        ret = np.bitwise_and(arr, np.bitwise_or(0XFE, v))
        return ret
    
    def getArrayLSB(arr: npt.NDArray) -> npt.NDArray:
        ret = np.zeros(arr.shape, dtype=arr.dtype)
        ret = np.bitwise_and(arr, 1)
        return ret

    # Modified version of: https://stackoverflow.com/questions/61094337/separating-2d-numpy-array-into-nxn-chunks
    def getImageChunks(data: npt.NDArray, chunk_dims: npt.NDArray) -> npt.NDArray:
        Y, X = chunk_dims

        A = []
        for v in np.array_split(data, data.shape[0] // Y, 0):
            A.extend([*np.array_split(v, data.shape[1] // X, 1)])
        A = np.array(A)
        A = np.split(A, data.shape[0] // Y)
        return np.array(A)
    
    def joinImageChunks(data: npt.NDArray) -> npt.NDArray:
        return np.hstack(np.hstack(data))

    # Operations
    @staticmethod
    def insert(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        image_data = modified['image']
        watermark = Watermark.watermark
        watermark_LSB = Watermark.getArrayLSB(watermark)

        image_blocks = Watermark.getImageChunks(image_data, watermark.shape) # Handle image in blocks

        for i in range(image_blocks.shape[0]):
            for j in range(image_blocks.shape[1]):
                block = image_blocks[i,j]
                block = Watermark.setArrayLSBTo(block, False)

                userKey = values[f'{Watermark.name()}_USER_KEY']
                imageId = values[f'{Watermark.name()}_IMAGE_ID']
                blockIndex = i * image_blocks.shape[1] + j
                hashBlock = Watermark.getHashBlock(userKey, imageId, image_data.shape, blockIndex, block, desiredBlockSize=block.size, watermarkShape=watermark.shape)

                Cr = np.bitwise_xor(hashBlock, watermark_LSB)
                Cr = np.bitwise_and(Cr, 1) # Limit to LSB

                # Insert Cr into LSB of image
                block = np.bitwise_or(block, Cr)

                image_blocks[i,j] = block # Store resulting block back

        modified['image'] = Watermark.joinImageChunks(image_blocks)

        return modified

    @staticmethod
    def extract(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        image_data = modified['image']
        image_blocks = Watermark.getImageChunks(image_data, Watermark.watermark.shape) # Handle image in blocks

        for i in range(image_blocks.shape[0]):
            for j in range(image_blocks.shape[1]):
                block = image_blocks[i,j]
                block_lsb = Watermark.getArrayLSB(block)


                block_zeroed = Watermark.setArrayLSBTo(block, False)

                userKey = values[f'{Watermark.name()}_USER_KEY']
                imageId = values[f'{Watermark.name()}_IMAGE_ID']
                blockIndex = i * image_blocks.shape[1] + j
                hashBlock = Watermark.getHashBlock(userKey, imageId, image_data.shape, blockIndex, block_zeroed, desiredBlockSize=block_zeroed.size, watermarkShape=Watermark.watermark.shape)

                Yr = Watermark.getArrayLSB(np.bitwise_xor(hashBlock, block_lsb))

                image_blocks[i,j] = Yr
        
        modified['image'] = Watermark.joinImageChunks(image_blocks)

        return modified
        
    
    @staticmethod
    def watermark_selected(original: dict[str, str], modified: dict[str, str], window, values) -> dict[str, str]:
        window[f'{Watermark.name()}_INSERT'].update(disabled=False) # Enable inserting watermark
        window[f'{Watermark.name()}_EXTRACT'].update(disabled=False) # Enable extracting watermark

        # Load watermark into binary image and store in static variable
        filename = values["WATERMARK_SELECTED"]
        
        input_image = Image.open(filename)      # Load image
        input_image = input_image.convert('1') # Convert to binary mage
        Watermark.watermark = np.asarray(input_image) # Extract pixel values as array


        return copy.deepcopy(modified)

    # Events
    @staticmethod
    def get_operation(operation_name: str) -> callable:
        if operation_name == f'{Watermark.name()}_INSERT':
            return Watermark.insert
        elif operation_name == f'{Watermark.name()}_EXTRACT':
            return Watermark.extract
        elif operation_name == f'{Watermark.name()}_SELECTED':
            return Watermark.watermark_selected
        else:
            print("Unknown watermark operation")
        return

def get_instance():
    return Watermark()
import PIL
from PIL import Image
import io
import base64

def convert_to_bytes(source, size=(None, None), subsample=None, zoom=None, fill=False):
    """
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param source: either a string filename or a bytes base64 image object
    :type source:  (Union[str, bytes])
    :param size:  optional new size (width, height)
    :type size: (Tuple[int, int] or None)
    :param subsample: change the size by multiplying width and height by 1/subsample
    :type subsample: (int)
    :param zoom: change the size by multiplying width and height by zoom
    :type zoom: (int)
    :param fill: If True then the image is filled/padded so that the image is square
    :type fill: (bool)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    """
    if isinstance(source, str):
        image = Image.open(source)
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(base64.b64decode(source)))
    else:
        image = PIL.Image.open(io.BytesIO(source))

    width, height = image.size

    scale = None
    if size != (None, None):
        new_width, new_height = size
        scale = min(new_height/height, new_width/width)
    elif subsample is not None:
        scale = 1/subsample
    elif zoom is not None:
        scale = zoom

    resized_image = image.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS) if scale is not None else image
    if fill and scale is not None:
        resized_image = make_square(resized_image)
    # encode a PNG formatted version of image into BASE64
    with io.BytesIO() as bio:
        resized_image.save(bio, format="PNG")
        contents = bio.getvalue()
        encoded = base64.b64encode(contents)
    return encoded

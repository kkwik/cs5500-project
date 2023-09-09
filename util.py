from PIL import Image
import io

def np_arr_to_byteio(data):
    img = Image.fromarray(data, mode="L")   # Treat data as 8-bit
    bio = io.BytesIO()  
    img.save(bio, format="PPM") # Save to object
    return bio.getvalue()
from PIL import Image
from io import BytesIO

image_file = []

img_bytes = BytesIO(image_file)
open_img = Image.open(img_bytes)
arr = np.array(open_img.resize((299,299)))
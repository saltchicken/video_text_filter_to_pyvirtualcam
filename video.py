import cv2
import os
import numpy as np
import pyvirtualcam
from PIL import Image, ImageFont, ImageDraw

########################################################################
# SETTINGS #
WIDTH = 1920//2
HEIGHT = 1080//2
FPS = 60
########################################################################

def convert_row_to_ascii(row):
    ORDER = (' ', '.', "'", ',', ':', ';', 'c', 'l',
             'x', 'o', 'k', 'X', 'd', 'O', '0', 'K', 'N')
    return tuple(ORDER[int(x / (255 / 16))] for x in row)[::-1]

def convert_to_ascii(input_grays):
    return tuple(convert_row_to_ascii(row) for row in input_grays)

## TODO Optimize camera resolution.
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS, print_fps=True) as cam:
    print(f'Using virtual camera: {cam.device}')

    ## Determine proper font size
    monospace = ImageFont.truetype("./Fonts/ANDALEMO.ttf",16)
    
    ## TODO Use textSize to optimize dimensions for ascii output
    text = "Test"
    textSize = monospace.getsize(text)
    
    frame_background = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while(cv2.waitKey(1) & 0xFF != ord('q')):

        # Get image data
        ret, frame = cap.read()

        # Convert data to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ## TODO Optimize dimensions
        # Reduce grayscale array to proper resolution
        reduced = cv2.resize(gray, (80, 45))
        
        # Make PIL image
        im_p = Image.fromarray(frame_background)

        # Get a drawing context
        draw = ImageDraw.Draw(im_p)
        
        ## TODO Optimize indexing.
        # Plug in reduced resolution numpy array for ascii converter func
        converted = convert_to_ascii(reduced)    
        for index, row in enumerate(converted):
            draw.text((0, index * 12),''.join(row),(255,255,255),font=monospace)

        # Convert back to OpenCV image and save
        result_o = np.array(im_p)

        cam.send(result_o)

cap.release()
cv2.destroyAllWindows()

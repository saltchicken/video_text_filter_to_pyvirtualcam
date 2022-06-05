import cv2
import os
import numpy as np
import pyvirtualcam
from PIL import Image, ImageFont, ImageDraw

########################################################################
# SETTINGS #
WIDTH = 640
HEIGHT = 360
FPS = 60
########################################################################

def convert_row_to_ascii(row):
    # 17-long
    ORDER = (' ', '.', "'", ',', ':', ';', 'c', 'l',
             'x', 'o', 'k', 'X', 'd', 'O', '0', 'K', 'N')
    return tuple(ORDER[int(x / (255 / 16))] for x in row)[::-1]

def convert_to_ascii(input_grays):
    return tuple(convert_row_to_ascii(row) for row in input_grays)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS, print_fps=True) as cam:
    print(f'Using virtual camera: {cam.device}')

    monospace = ImageFont.truetype("./Fonts/ANDALEMO.ttf",32)
    text = "Test"
    textSize = monospace.getsize(text)
    
    frame_blank = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while(cv2.waitKey(1) & 0xFF != ord('q')):

        # Get image data
        ret, frame = cap.read()

        ## Convert data to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ## Reduce grayscale array to proper resolution
        reduced = cv2.resize(gray, (int(cam.width), int(cam.height)))
        # cv2.imshow('frame', reduced)
        
		# Make PIL image
        im_p = Image.fromarray(frame_blank)

		# Get a drawing context
        draw = ImageDraw.Draw(im_p)
		
        ## Plug in reduced resolution numpy array for ascii converter func
        converted = convert_to_ascii(reduced)    
		
		## TODO use `converted` as text input
        draw.text((40, 80),text,(255,255,255),font=monospace)

		# Convert back to OpenCV image and save
        result_o = np.array(im_p)

        cv2.imshow('frame', result_o)
            
        # frame_blank[:] = cam.frames_sent % 255  # grayscale animation
        # cam.send(frame_blank)
        # cam.sleep_until_next_frame()

cap.release()
cv2.destroyAllWindows()

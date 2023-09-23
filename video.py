import cv2
import os
import numpy as np
import pyvirtualcam
from PIL import Image, ImageFont, ImageDraw
import multiprocessing

########################################################################
# SETTINGS #
WIDTH = 1920
HEIGHT = 1080
FPS = 60

RESIZED_WIDTH = 160 # 240, 112, 80
RESIZED_HEIGHT = 90 # 135, 63, 45
########################################################################

def convert_row_to_ascii(row):
    ORDER = (' ', '.', "'", ',', ':', ';', 'c', 'l',
             'x', 'o', 'k', 'X', 'd', 'O', '0', 'K', 'N')
    return tuple(ORDER[int(x / (255 / 16))] for x in row)[::-1]

def convert_to_ascii(input_grays):
    return tuple(convert_row_to_ascii(row) for row in input_grays)

def worker(queue_input, queue_output):
    monospace = ImageFont.truetype("./Fonts/ANDALEMO.ttf",20)
    frame_background = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    while True:
        reduced = queue_input.get()
        if reduced is None:
            # Poison pill to signal the worker process to exit.
            break
        im_p = Image.fromarray(frame_background)

        # Get a drawing context
        draw = ImageDraw.Draw(im_p)
        
        ## TODO Optimize indexing.
        # Plug in reduced resolution numpy array for ascii converter func
        converted = convert_to_ascii(reduced)    
        for index, row in enumerate(converted):
            draw.text((0, index * (HEIGHT / RESIZED_HEIGHT)),''.join(row),(0,255,0),font=monospace)

        # Convert back to OpenCV image and save
        result_o = np.array(im_p)
        queue_output.put(result_o)
        
def sender(queue_output):
    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS, print_fps=True) as cam:
        print(f'Using virtual camera: {cam.device}')
        while True:
            result_o = queue_output.get()
            if result_o is None:
                # Poison pill
                break
            cam.send(result_o)
    
    
if __name__ == "__main__":
    # print(f"Number of cores: {multiprocessing.cpu_count()}")
    ## TODO Optimize camera resolution.
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


    queue_input = multiprocessing.Queue()
    queue_output = multiprocessing.Queue()
    
    pool = multiprocessing.Pool(8)

    for i in range(16):
        p = multiprocessing.Process(target=worker, args=(queue_input, queue_output))
        p.start()
        
    process = multiprocessing.Process(target=sender, args=(queue_output,))
    process.start()
    
    while(cv2.waitKey(1) & 0xFF != ord('q')):

        # Get image data
        ret, frame = cap.read()

        # Convert data to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ## TODO Optimize dimensions
        # Reduce grayscale array to proper resolution
        reduced = cv2.resize(gray, (RESIZED_WIDTH, RESIZED_HEIGHT))
        queue_input.put(reduced)

# TODO: Insert a proper break to the while loop so that these following functions actually run. Then send 16 (number of pool) None to queue_input and 1 to queue_output
    cap.release()
    cv2.destroyAllWindows()
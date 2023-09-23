import cv2
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

ROW_SPACING = HEIGHT / RESIZED_HEIGHT

LOWER_BLUE = np.array([100, 0, 0])
UPPER_BLUE = np.array([255, 100, 120])

CAP_INPUT = 0 # 0 for camera
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
            # Poison pill
            break
        im_p = Image.fromarray(frame_background)
        draw = ImageDraw.Draw(im_p)
        
        # Plug in reduced resolution numpy array for ascii converter func
        converted = convert_to_ascii(reduced)    
        for index, row in enumerate(converted):
            # TODO: This can be improved with multiline drawing
            draw.text((0, index * (ROW_SPACING)),''.join(row),(0,255,0),font=monospace)

        # Convert back to OpenCV image
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
    ## TODO Optimize camera resolution.
    cap = cv2.VideoCapture(CAP_INPUT)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    queue_input = multiprocessing.Queue()
    queue_output = multiprocessing.Queue()

    available_cores = multiprocessing.cpu_count()
    process_cores = 11 if available_cores >= 16 else available_cores - 3
    assert process_cores >= 4
    for i in range(process_cores):
        p = multiprocessing.Process(target=worker, args=(queue_input, queue_output))
        p.start()
        
    process = multiprocessing.Process(target=sender, args=(queue_output,))
    process.start()
    
    while(cv2.waitKey(1) & 0xFF != ord('q')):
        try:
            ret, frame = cap.read()
            image = cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT))
            mask = cv2.inRange(image, LOWER_BLUE, UPPER_BLUE)
            image[mask != 0] = [0, 0, 0]
            reduced = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            queue_input.put(reduced)
        except Exception as e:
            print(e)
            break
        
    for i in range(process_cores):
        queue_input.put(None)
    queue_output.put(None)
    cap.release()
    cv2.destroyAllWindows()
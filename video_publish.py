import cv2
import numpy as np
import pyvirtualcam
from PIL import Image, ImageFont, ImageDraw
import multiprocessing, time
import zmq

########################################################################
# SETTINGS #
WIDTH = 1920
HEIGHT = 1080
FPS = 60

RESIZED_WIDTH = 240     # 256, 240, 160, 112, 80
RESIZED_HEIGHT = 135     # 144, 135, 90, 63, 45
FONT_SIZE = 14      # 14, 14, 20, 28, 40
# FONT_ALPHA = 1         # 1, 1, 2, 3, 3
ROW_SPACING = HEIGHT / RESIZED_HEIGHT

LOWER_BLUE = np.array([100, 0, 0])
UPPER_BLUE = np.array([255, 100, 120])

CAP_INPUT = "C:/Users/johne/Desktop/sample2.mkv" # 0 for camera
########################################################################

def convert_row_to_ascii(row):
    ORDER = (' ', '.', "'", ',', ':', ';', 'c', 'l',
             'x', 'o', 'k', 'X', 'd', 'O', '0', 'K', 'N')
    return tuple(ORDER[int(x / (255 / 16))] for x in row)[::-1]

def convert_to_ascii(input_grays):
    return tuple(convert_row_to_ascii(row) for row in input_grays)

def worker(queue_input, queue_output):
    monospace = ImageFont.truetype("./Fonts/ANDALEMO.ttf", FONT_SIZE)
    # _, top, _, bottom = monospace.getbbox(" .',:;clxokXdO0KN")
    # SPACING = ROW_SPACING - (bottom - top) - FONT_ALPHA
    frame_background = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    context = zmq.Context()
    puller = context.socket(zmq.PULL)
    puller.connect("tcp://127.0.0.1:5559")
    context = zmq.Context()
    pusher = context.socket(zmq.PUSH)
    pusher.connect("tcp://127.0.0.1:5558") 
    while True:
        try:
            # try:
            #     reduced = puller.recv(flags=zmq.NOBLOCK)
            # except zmq.Again as e:
            #     continue
            # except KeyboardInterrupt:
            #     break
            reduced = puller.recv()
            deserialized_image = np.frombuffer(reduced, dtype=np.uint8)
            deserialized_image = deserialized_image.reshape(RESIZED_HEIGHT, RESIZED_WIDTH)

            im_p = Image.fromarray(frame_background)
            draw = ImageDraw.Draw(im_p)
            
            # Plug in reduced resolution numpy array for ascii converter func

            converted = convert_to_ascii(deserialized_image)
            # converted = convert_to_ascii(reduced)
            
            # multiline_string = ''
            # for row in converted:
            #     multiline_string += ''.join(row) + '\n'
            # draw.multiline_text((0,0), multiline_string, (0,255,0), spacing=SPACING, font=monospace)
            
            
            for index, row in enumerate(converted):
                draw.text((0, index * (ROW_SPACING)),''.join(row),(0,255,0),font=monospace)
            
            # Convert back to OpenCV image
            result_o = np.array(im_p)
            received_data = np.frombuffer(result_o, dtype=np.int32)
            serialized_bytes = received_data.tobytes()
            pusher.send(serialized_bytes, flags=zmq.NOBLOCK)

        except Exception as e:
            print(e)
        
def sender(queue_output):
    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS, print_fps=True) as cam:
        print(f'Using virtual camera: {cam.device}')
        context = zmq.Context()
        puller = context.socket(zmq.PULL)
        puller.bind("tcp://127.0.0.1:5558")
        while True:
            try:
                # try:
                #     result_o = puller.recv(flags=zmq.NOBLOCK)
                # except zmq.Again as e:
                #     time.sleep(0.01)
                #     continue
                # except KeyboardInterrupt:
                #     break
                result_o = puller.recv()
                deserialized_image = np.frombuffer(result_o, dtype=np.uint8)
                deserialized_image = deserialized_image.reshape(HEIGHT, WIDTH, 3)
                cam.send(deserialized_image)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    context = zmq.Context()
    pusher = context.socket(zmq.PUSH)
    pusher.bind("tcp://*:5559")  # Publisher binds to a specific address and port
    
    cap = cv2.VideoCapture(CAP_INPUT)
    cap_width  = cap.get(3)
    cap_height = cap.get(4)
    
    queue_input = multiprocessing.Queue()
    queue_output = multiprocessing.Queue()
    
    process = multiprocessing.Process(target=sender, args=(queue_output,))
    process.start()

    available_cores = multiprocessing.cpu_count()
    process_cores = 21 if available_cores >= 24 else available_cores - 3
    assert process_cores >= 4
    for i in range(process_cores):
        p = multiprocessing.Process(target=worker, args=(queue_input, queue_output))
        p.start()
    
    # TODO: Gives time for process to load up. This prevents the hyper FPS when starting from video. This can
    # be resolved by having the process running as a standalone service.
    time.sleep(2)
    
    while(cv2.waitKey(1) & 0xFF != ord('q')):
        try:
            ret, frame = cap.read()
            # This will loop the source if reading from a file.
            if not ret:
                cap = cv2.VideoCapture(CAP_INPUT)
                continue
            if frame is None:
                continue
            
            image = cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT))
            mask = cv2.inRange(image, LOWER_BLUE, UPPER_BLUE)
            image[mask != 0] = [0, 0, 0]
            reduced = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # print("shape!!!!!!!!!!!!!!!!!!!!!:", reduced.shape)
            serialized_bytes = reduced.tobytes()
            pusher.send(serialized_bytes)
            # queue_input.put(reduced)
        except Exception as e:
            print(e)
            break
        
    # for i in range(process_cores):
    #     queue_input.put(None)
    # queue_output.put(None)
    cap.release()
    cv2.destroyAllWindows()
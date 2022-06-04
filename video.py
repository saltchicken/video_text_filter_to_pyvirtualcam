import cv2
import os
import numpy as np
import pyvirtualcam

########################################################################
# SETTINGS #
WIDTH = 1280
HEIGHT = 720
FPS = 20
########################################################################


def convert_row_to_ascii(row):
    # 17-long
    ORDER = (' ', '.', "'", ',', ':', ';', 'c', 'l',
             'x', 'o', 'k', 'X', 'd', 'O', '0', 'K', 'N')
    return tuple(ORDER[int(x / (255 / 16))] for x in row)[::-1]


def convert_to_ascii(input_grays):
    return tuple(convert_row_to_ascii(row) for row in input_grays)

cap = cv2.VideoCapture(0)

with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS, print_fps=True) as cam:
	print(f'Using virtual camera: {cam.device}')
	while(cv2.waitKey(1) & 0xFF != ord('q')):
		screen_height, screen_width = height, width

		# Get image data
		ret, frame = cap.read()

		# Convert data to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Reduce grayscale array to proper resolution
		reduced = cv2.resize(gray, (int(screen_width), int(screen_height)))

		# Plug in reduced resolution numpy array for ascii converter func
		converted = convert_to_ascii(reduced)
			
		frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
		frame[:] = cam.frames_sent % 255  # grayscale animation
		cam.send(frame)
		cam.sleep_until_next_frame()


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

import zmq
import time
import numpy as np

context = zmq.Context()
publisher = context.socket(zmq.PUSH)
publisher.bind("tcp://*:5555")  # Publisher binds to a specific address and port

while True:
    try:
        # start_time = str(time.time_ns())
        message = np.array([1,2,3,4,5]).tobytes()
        # publisher.send_string(message)
        publisher.send(message)
        time.sleep(0.5)  # Publish a message every second
    except Exception as e:
        print(e)
import zmq
import time
import numpy as np

context = zmq.Context()
puller = context.socket(zmq.PULL)
puller.connect("tcp://192.168.1.13:5556")  # Connect to the publisher's address and port

while True:
    message = puller.recv()
    received_data = np.frombuffer(message, dtype=np.int32) 
    print(f' Message: {received_data}')
    # duration = time.time_ns() - int(message)
    # print(f"Duration: {duration}")

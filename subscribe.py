import zmq
import numpy as np

context = zmq.Context()
puller = context.socket(zmq.PULL)
puller.connect("tcp://192.168.1.10:5555")  # Connect to the publisher's address and port

context = zmq.Context()
publisher = context.socket(zmq.PUSH)
publisher.bind("tcp://*:5556")  # Publisher binds to a specific address and port

while True:
    try:
        message = puller.recv()
        received_data = np.frombuffer(message, dtype=np.int32) 
        print(f"Received Inter: {message}")
        publisher.send(message)
    except Exception as e:
        print(e)

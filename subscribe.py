import zmq

context = zmq.Context()
puller = context.socket(zmq.PULL)
puller.connect("tcp://192.168.1.10:5555")  # Connect to the publisher's address and port

while True:
    message = puller.recv_string()
    print(f"Received: {message}")

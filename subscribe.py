import zmq

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://192.168.1.10:5555")  # Connect to the publisher's address and port

# Subscribe to all messages (empty string means subscribe to all topics)
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

while True:
    message = subscriber.recv_string()
    print(f"Received: {message}")

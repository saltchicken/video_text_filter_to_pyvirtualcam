import zmq
import time

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:5555")  # Publisher binds to a specific address and port

while True:
    message = "Trask Distribution Message"
    print(f"Publishing: {message}")
    publisher.send_string(message)
    time.sleep(1)  # Publish a message every second

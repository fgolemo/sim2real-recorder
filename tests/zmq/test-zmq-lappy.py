import time

import zmq

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://flogo3.local:%s" % port)

# msg = socket.recv()
# print msg

while True:
    msg = raw_input("type msg:")
    socket.send_string(msg)
    time.sleep(1)

import time
import zmq

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:%s" % port)

socket.send("welcome from ergo jr")
while True:
    msg = socket.recv()
    print (msg)
    time.sleep(1)

import numpy as np
import zmq


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://flogo3.local:%s" % port)

msg = socket.recv()
print msg

# while True:
#     msg = raw_input("type msg:")
#     socket.send_string(msg)
#     time.sleep(1)

socket.send_string("abc hi")

socket.send_string("numpy")
send_array(socket, np.array([[1, 2], [3, 4]], dtype=np.uint8))

socket.send_string("666") # will trigger np sending
a = recv_array(socket)
print (a)

socket.send_string("quit")

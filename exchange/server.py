import numpy
import zmq


class Server():
    def __init__(self, port="5556"):
        self.port = port
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % port)
        self.cb = self.noop

    def noop(self, msg):
        pass

    def welcome(self):
        self.socket.send("welcome from ergo jr")

    def register_callback(self, cb):
        self.cb = cb

    def start_main_loop(self):
        while True:
            msg = self.socket.recv_string()
            if (msg == "quit"):
                print ("got quit signal... exiting.")
                break
            elif (msg == "numpy"):
                msg = self.recv_array()

            self.cb(msg)

    def send_array(self, A, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.socket.send_json(md, flags | zmq.SNDMORE)
        return self.socket.send(A, flags, copy=copy, track=track)

    def recv_array(self, flags=0, copy=True, track=False):
        """recv a numpy array"""
        md = self.socket.recv_json(flags=flags)
        msg = self.socket.recv(flags=flags, copy=copy, track=track)
        buf = buffer(msg)
        A = numpy.frombuffer(buf, dtype=md['dtype'])
        return A.reshape(md['shape'])

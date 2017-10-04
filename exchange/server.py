import numpy
import zmq

from exchange.utilities import zmq_recv_array, zmq_send_array


class Server():
    def __init__(self, port="5556"):
        self.port = port
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % port)
        self.cb = self.noop

    def noop(self, msg, send):
        pass

    def welcome(self):
        self.socket.send("welcome from ergo jr", zmq.NOBLOCK)

    def register_callback(self, cb):
        self.cb = cb

    def start_main_loop(self):
        print ("STARTING MAIN SERVER LOOP")
        while True:
            msg = self.socket.recv_string()
            if (msg == "quit"):
                print ("got quit signal... exiting.")
                break
            elif (msg == "numpy"):
                msg = self.recv_array()

            self.cb(msg, self.send_array)

    def send_array(self, A, flags=0, copy=True, track=False):
        return zmq_send_array(self.socket, A, flags, copy, track)

    def recv_array(self, flags=0, copy=True, track=False):
        return zmq_recv_array(self.socket, flags, copy, track)

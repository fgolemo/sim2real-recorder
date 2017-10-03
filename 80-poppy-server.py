from exchange.server import Server
import numpy as np

server = Server()

server.welcome()

ladida = "test:"

def handle_message(msg, send):
    print(ladida, msg)
    if msg == "666":
        send(np.array([1,2,3]))

server.register_callback(handle_message)
server.start_main_loop()

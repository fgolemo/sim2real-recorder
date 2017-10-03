from exchange.server import Server

server = Server()

server.welcome()

ladida = "test:"


def handle_message(msg):
    print(ladida, msg)


server.register_callback(handle_message)
server.start_main_loop()

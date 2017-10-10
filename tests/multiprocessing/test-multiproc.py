from multiprocessing import Process, Queue

import time


def f(q):
    for i in range(8):
        q.put([i, 42, None, 'hello'])
        time.sleep(.5)

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    time.sleep(3)
    print q.qsize()
    last_out = q.get()
    while not q.empty():
        last_out = q.get()
    print last_out    # prints "[42, None, 'hello']"
    p.join()
